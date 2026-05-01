"""
Experiment 4.1: Human vs AI
  !python /content/Evaluation-of-AI-Generated-Text-Detection-Methods/run_detectgpt_on_datasets.py \
    --human_csv /content/Evaluation-of-AI-Generated-Text-Detection-Methods/data/processed/plain_human.csv \
    --ai_csv /content/Evaluation-of-AI-Generated-Text-Detection-Methods/data/processed/plain_ai.csv \
    --experiment_name human_vs_plain_ai \
    --n_samples 100 \
    --n_perturbations 50 \
    --base_model_name gpt2 \
    --generator_model gpt2
      

Experiment 4.2: Human vs Paraphrased AI
  !python run_detectgpt_on_datasets.py \
      --human_csv data/processed/plain_human.csv \
      --ai_csv data/processed/paraphrased_ai.csv \
      --experiment_name human_vs_paraphrased \
      --n_samples 100 \
      --n_perturbations 50 \
      --base_model_name gpt2

Experiment 4.3: Human vs Watermarked AI
  !python run_detectgpt_on_datasets.py \
      --human_csv data/processed/plain_human.csv \
      --ai_csv data/processed/watermarked_ai_100.csv \
      --experiment_name human_vs_watermarked \
      --n_samples 100 \
      --n_perturbations 50 \
      --base_model_name gpt2 \
      --generator_model gpt2
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
from sklearn.metrics import roc_auc_score, roc_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model Loading

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


# DetectGPT core from original repo

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
            **inputs, max_length=512, do_sample=True, top_p=0.96
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
        if len(pt.strip().split()) > 10 and not re.search(r'<extra_id_\d+>', pt):
            p_lls.append(get_ll(pt, base_model, base_tokenizer))


    if len(p_lls) < 2:
        return 0.0, 0.0

    mean_p = np.mean(p_lls)
    std_p = np.std(p_lls) + 1e-10
    d = original_ll - mean_p
    z = d / std_p
    return d, z


# Data Loading

def load_subset(csv_path, n_samples=None, min_words=30, generator_model=None):
    """Load a processed CSV from data/processed/."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError(f"CSV must have a 'text' column. Found: {df.columns.tolist()}")

    df = df[df["text"].notna()].copy()
    df = df[df["text"].str.split().str.len() >= min_words].copy()

    if generator_model and "generator_model" in df.columns:
        df = df[df["generator_model"] == generator_model].copy()
        print(f"  Filtered to generator_model='{generator_model}': {len(df)} rows")

    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    print(f"  Loaded {len(df)} samples (min {min_words} words)")

    if "generator_model" in df.columns:
        print(f"  Models: {df['generator_model'].value_counts().head(5).to_dict()}")
    if "domain" in df.columns:
        print(f"  Domains: {df['domain'].value_counts().head(5).to_dict()}")

    return df["text"].tolist()


# Evaluation

def compute_clf_metrics(labels, scores):
    auroc = roc_auc_score(labels, scores)
    if auroc < 0.5:
        scores = [-s for s in scores]
        auroc = 1 - auroc

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    fpr_at_95_tpr = float(fpr[min(idx, len(fpr) - 1)])

    return {
        "auroc":         round(auroc, 4),
        "fpr_at_95_tpr": round(fpr_at_95_tpr, 4),
    }, scores


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

    # --- Compute metrics for each scorer ---
    ll_metrics, ll_oriented = compute_clf_metrics(labels, ll_scores)
    d_metrics,  d_oriented  = compute_clf_metrics(labels, d_scores)
    z_metrics,  z_oriented  = compute_clf_metrics(labels, z_scores)

    n_samples_dataset1 = len(human_texts)
    n_samples_dataset2 = len(ai_texts)

    results = {
        "log_likelihood": {
            **ll_metrics,
            "n_samples_dataset1": n_samples_dataset1,
            "n_samples_dataset2": n_samples_dataset2,
            "y_true":  labels,
            "y_score": ll_oriented,
        },
        "detectgpt_d": {
            **d_metrics,
            "n_samples_dataset1": n_samples_dataset1,
            "n_samples_dataset2": n_samples_dataset2,
            "y_true":  labels,
            "y_score": d_oriented,
        },
        "detectgpt_z": {
            **z_metrics,
            "n_samples_dataset1": n_samples_dataset1,
            "n_samples_dataset2": n_samples_dataset2,
            "y_true":  labels,
            "y_score": z_oriented,
        },
    }

    for scorer_name, scorer_results in results.items():
        print(f"\nEvaluation Results ({scorer_name}):")
        for k, v in scorer_results.items():
            if k not in ("y_true", "y_score"):
                print(f"  {k}: {v}")

    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# Main

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
    p.add_argument("--n_perturbations", type=int, default=50,
                   help="Number of T5 perturbations per text")
    p.add_argument("--base_model_name", default="gpt2",
                   help="Model that generated the AI-text (default: gpt2)")
    p.add_argument("--generator_model", default="gpt2",
                   help="Filter AI CSV rows to this generator_model value (default: gpt2)")
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
    ai_texts = load_subset(args.ai_csv, n_samples=args.n_samples, generator_model=args.generator_model)

    # Run
    run_experiment(
        human_texts, ai_texts,
        base_model, base_tok, mask_model, mask_tok,
        n_perturbations=args.n_perturbations,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
