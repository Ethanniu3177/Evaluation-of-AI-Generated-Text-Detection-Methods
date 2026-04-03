"""
run_detectgpt_on_datasets.py

Runs DetectGPT scoring on the processed CSV files produced by build_datasets.py.
Place this file in your project root (same level as build_datasets.py).

Your data/processed/ folder should contain:
  - plain_human.csv
  - plain_ai.csv
  - paraphrased_ai.csv
  - watermarked_ai_<N>.csv

Usage (in Colab):
  # Experiment 4.1: Human vs AI
  !python run_detectgpt_on_datasets.py \
      --human_csv data/processed/plain_human.csv \
      --ai_csv data/processed/plain_ai.csv \
      --experiment_name human_vs_ai \
      --n_samples 100

  # Experiment 4.2: Human vs Paraphrased AI
  !python run_detectgpt_on_datasets.py \
      --human_csv data/processed/plain_human.csv \
      --ai_csv data/processed/paraphrased_ai.csv \
      --experiment_name human_vs_paraphrased

  # Experiment 4.3: Human vs Watermarked AI
  !python run_detectgpt_on_datasets.py \
      --human_csv data/processed/plain_human.csv \
      --ai_csv data/processed/watermarked_ai_100.csv \
      --experiment_name human_vs_watermarked
"""

import argparse
import os
import json
import numpy as np
import torch
import transformers
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL LOADING
# ============================================================

def load_base_model(name, cache_dir):
    print(f"Loading base model {name}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir
    ).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def load_mask_model(name, cache_dir):
    print(f"Loading mask model {name}...")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        name, cache_dir=cache_dir
    ).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, model_max_length=512, cache_dir=cache_dir
    )
    model.eval()
    return model, tokenizer


# ============================================================
# DETECTGPT CORE (extracted + cleaned from eric-mitchell/detect-gpt)
# ============================================================

def get_ll(text, model, tokenizer, max_length=512):
    """Log-likelihood of text under a causal LM."""
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(DEVICE)
        output = model(**inputs, labels=inputs.input_ids)
        return -output.loss.item()


def tokenize_and_mask(text, span_length=2, pct=0.3, buffer_size=1):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    n_spans = max(1, int(pct * len(tokens) / (span_length + buffer_size * 2)))
    n_masks = 0
    attempts = 0
    while n_masks < n_spans and attempts < n_spans * 10:
        start = np.random.randint(0, max(1, len(tokens) - span_length))
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
        attempts += 1

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    return ' '.join(tokens)


def replace_masks(texts, mask_model, mask_tokenizer):
    with torch.no_grad():
        inputs = mask_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(DEVICE)
        outputs = mask_model.generate(
            **inputs, max_length=150, do_sample=True, top_p=0.96
        )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    extracted = []
    for text in texts:
        pattern = re.compile(r"<extra_id_\d+>(.*?)(?=<extra_id_\d+>|</s>|$)")
        matches = pattern.findall(text)
        extracted.append([m.strip() for m in matches])
    return extracted


def apply_extracted_fills(masked_texts, extracted_fills):
    results = []
    for masked, fills in zip(masked_texts, extracted_fills):
        tokens = masked.split(' ')
        fill_idx = 0
        result = []
        for token in tokens:
            if re.match(r'<extra_id_\d+>', token) and fill_idx < len(fills):
                result.append(fills[fill_idx])
                fill_idx += 1
            else:
                result.append(token)
        results.append(' '.join(result))
    return results


def perturb_texts(texts, mask_model, mask_tokenizer, span_length=2, pct=0.3,
                  chunk_size=10):
    masked = [tokenize_and_mask(t, span_length, pct) for t in texts]
    all_perturbed = []
    for i in range(0, len(masked), chunk_size):
        chunk = masked[i:i + chunk_size]
        raw_fills = replace_masks(chunk, mask_model, mask_tokenizer)
        extracted = extract_fills(raw_fills)
        perturbed = apply_extracted_fills(chunk[:len(extracted)], extracted)
        all_perturbed.extend(perturbed)
    return all_perturbed


def detectgpt_score(text, base_model, base_tokenizer, mask_model,
                    mask_tokenizer, n_perturbations=10, span_length=2, pct=0.3):
    """
    Returns (d_score, z_score).
    Higher = more likely machine-generated.
    """
    original_ll = get_ll(text, base_model, base_tokenizer)

    copies = [text] * n_perturbations
    perturbed = perturb_texts(copies, mask_model, mask_tokenizer,
                              span_length=span_length, pct=pct)

    p_lls = []
    for pt in perturbed:
        if len(pt.strip().split()) > 10:
            p_lls.append(get_ll(pt, base_model, base_tokenizer))

    if len(p_lls) < 2:
        return 0.0, 0.0

    mean_p = np.mean(p_lls)
    std_p = np.std(p_lls) + 1e-10
    d = original_ll - mean_p
    z = d / std_p
    return d, z


# ============================================================
# DATA LOADING — reads YOUR build_datasets.py output
# ============================================================

def load_subset(csv_path, n_samples=None, min_words=30):
    """Load a processed CSV from data/processed/."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter: need a 'text' column with enough content
    if "text" not in df.columns:
        raise ValueError(f"CSV must have a 'text' column. Found: {df.columns.tolist()}")

    df = df[df["text"].notna()].copy()
    df = df[df["text"].str.split().str.len() >= min_words].copy()

    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    print(f"  Loaded {len(df)} samples (min {min_words} words)")

    # Print domain/model breakdown if columns exist
    if "generator_model" in df.columns:
        print(f"  Models: {df['generator_model'].value_counts().head(5).to_dict()}")
    if "domain" in df.columns:
        print(f"  Domains: {df['domain'].value_counts().head(5).to_dict()}")

    return df["text"].tolist()


# ============================================================
# EVALUATION
# ============================================================

def run_experiment(human_texts, ai_texts, base_model, base_tokenizer,
                   mask_model, mask_tokenizer, n_perturbations=10,
                   output_dir="eval_results"):
    os.makedirs(output_dir, exist_ok=True)

    all_texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    # --- Log-likelihood baseline (fast) ---
    print(f"\n--- Log-likelihood baseline ({len(all_texts)} texts) ---")
    ll_scores = []
    for text in tqdm(all_texts, desc="Log-likelihood"):
        ll_scores.append(get_ll(text, base_model, base_tokenizer))

    ll_auroc = roc_auc_score(labels, ll_scores)
    if ll_auroc < 0.5:
        ll_scores = [-s for s in ll_scores]
        ll_auroc = 1 - ll_auroc
    print(f"Log-likelihood AUROC: {ll_auroc:.4f}")

    # --- DetectGPT (slow) ---
    print(f"\n--- DetectGPT ({n_perturbations} perturbations per text) ---")
    d_scores, z_scores = [], []
    for text in tqdm(all_texts, desc="DetectGPT"):
        d, z = detectgpt_score(
            text, base_model, base_tokenizer,
            mask_model, mask_tokenizer,
            n_perturbations=n_perturbations
        )
        d_scores.append(d)
        z_scores.append(z)

    d_auroc = roc_auc_score(labels, d_scores)
    z_auroc = roc_auc_score(labels, z_scores)

    print(f"\n{'='*55}")
    print(f"  Log-likelihood baseline  AUROC: {ll_auroc:.4f}")
    print(f"  DetectGPT (d, unnorm.)   AUROC: {d_auroc:.4f}")
    print(f"  DetectGPT (z, norm.)     AUROC: {z_auroc:.4f}")
    print(f"{'='*55}")

    # --- Save everything ---
    results = {
        "n_human": len(human_texts),
        "n_ai": len(ai_texts),
        "n_perturbations": n_perturbations,
        "ll_auroc": round(ll_auroc, 4),
        "detectgpt_d_auroc": round(d_auroc, 4),
        "detectgpt_z_auroc": round(z_auroc, 4),
        "scores": {
            "ll": ll_scores,
            "d": d_scores,
            "z": z_scores,
            "labels": labels,
        }
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- ROC plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for scores, name in [
        (d_scores, f"DetectGPT-d (AUC={d_auroc:.3f})"),
        (z_scores, f"DetectGPT-z (AUC={z_auroc:.3f})"),
        (ll_scores, f"Log-likelihood (AUC={ll_auroc:.3f})"),
    ]:
        s = scores
        a = roc_auc_score(labels, s)
        if a < 0.5:
            s = [-x for x in s]
        fpr, tpr, _ = roc_curve(labels, s)
        ax.plot(fpr, tpr, label=name)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("DetectGPT on Project Dataset")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(output_dir, "roc.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nResults saved to {output_dir}/")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Run DetectGPT on your processed datasets"
    )
    p.add_argument("--human_csv", required=True,
                   help="Path to plain_human.csv")
    p.add_argument("--ai_csv", required=True,
                   help="Path to AI text CSV (plain_ai, paraphrased_ai, or watermarked_ai)")
    p.add_argument("--experiment_name", default="experiment",
                   help="Name for the output folder")
    p.add_argument("--n_samples", type=int, default=100,
                   help="Max samples per class (human and AI)")
    p.add_argument("--n_perturbations", type=int, default=10,
                   help="Number of T5 perturbations per text")
    p.add_argument("--base_model_name", default="gpt2-medium")
    p.add_argument("--mask_model_name", default="t5-large")
    p.add_argument("--cache_dir", default="/root/.cache")
    p.add_argument("--output_base", default="eval_results",
                   help="Base output directory")

    args = p.parse_args()

    output_dir = os.path.join(args.output_base, args.experiment_name)

    # Load models
    base_model, base_tok = load_base_model(args.base_model_name, args.cache_dir)
    mask_model, mask_tok = load_mask_model(args.mask_model_name, args.cache_dir)

    # Load data from your CSVs
    human_texts = load_subset(args.human_csv, n_samples=args.n_samples)
    ai_texts = load_subset(args.ai_csv, n_samples=args.n_samples)

    # Run
    run_experiment(
        human_texts, ai_texts,
        base_model, base_tok, mask_model, mask_tok,
        n_perturbations=args.n_perturbations,
        output_dir=output_dir,
    )

    print("\nDone. Don't forget to copy results to Google Drive!")
    print(f"  !cp -r {args.output_base}/ /content/drive/MyDrive/detectgpt_eval/")


if __name__ == "__main__":
    main()
