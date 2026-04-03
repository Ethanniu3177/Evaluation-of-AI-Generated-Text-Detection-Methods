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
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    f1_score, recall_score, precision_score, accuracy_score,
)
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

def compute_clf_metrics(labels, scores):
    """
    Compute F1, recall, precision, accuracy, and AUROC for a scorer.
    Classification threshold is chosen to maximise Youden's J (TPR - FPR).
    Returns a dict of rounded metric values and the oriented scores.
    """
    auroc = roc_auc_score(labels, scores)
    # Orient so that higher score => AI (label=1)
    if auroc < 0.5:
        scores = [-s for s in scores]
        auroc = 1 - auroc

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = int(np.argmax(tpr - fpr))
    threshold = thresholds[best_idx]

    preds = [1 if s >= threshold else 0 for s in scores]
    return {
        "auroc":     round(auroc, 4),
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "threshold": round(float(threshold), 6),
    }, scores  # return oriented scores for plotting


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

    # --- Compute all 5 metrics for each scorer ---
    ll_metrics, ll_scores   = compute_clf_metrics(labels, ll_scores)
    d_metrics,  d_scores    = compute_clf_metrics(labels, d_scores)
    z_metrics,  z_scores    = compute_clf_metrics(labels, z_scores)

    print(f"\n{'='*65}")
    print(f"{'Metric':<12} {'Log-likelihood':>16} {'DetectGPT-d':>14} {'DetectGPT-z':>14}")
    print(f"{'-'*65}")
    for key in ("auroc", "f1", "recall", "precision", "accuracy"):
        print(f"{key:<12} {ll_metrics[key]:>16.4f} {d_metrics[key]:>14.4f} {z_metrics[key]:>14.4f}")
    print(f"{'='*65}")

    # --- Save everything ---
    results = {
        "n_human": len(human_texts),
        "n_ai": len(ai_texts),
        "n_perturbations": n_perturbations,
        "metrics": {
            "log_likelihood": ll_metrics,
            "detectgpt_d":    d_metrics,
            "detectgpt_z":    z_metrics,
        },
        "scores": {
            "ll":     ll_scores,
            "d":      d_scores,
            "z":      z_scores,
            "labels": labels,
        },
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Plot 1: ROC curves ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for scores, name, metrics in [
        (ll_scores, "Log-likelihood",  ll_metrics),
        (d_scores,  "DetectGPT-d",     d_metrics),
        (z_scores,  "DetectGPT-z",     z_metrics),
    ]:
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={metrics['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — DetectGPT on Project Dataset")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(output_dir, "roc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Precision-Recall curves ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for scores, name in [
        (ll_scores, "Log-likelihood"),
        (d_scores,  "DetectGPT-d"),
        (z_scores,  "DetectGPT-z"),
    ]:
        prec, rec, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})")
    baseline = sum(labels) / len(labels)
    ax.axhline(baseline, color='k', linestyle='--', alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — DetectGPT on Project Dataset")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Bar chart of all 5 metrics ----
    metric_keys  = ["auroc", "f1", "recall", "precision", "accuracy"]
    scorer_names = ["Log-likelihood", "DetectGPT-d", "DetectGPT-z"]
    scorer_data  = [ll_metrics, d_metrics, z_metrics]

    x = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, data) in enumerate(zip(scorer_names, scorer_data)):
        vals = [data[k] for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=name)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics — DetectGPT on Project Dataset")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {output_dir}/")
    print(f"  results.json, roc.png, pr_curve.png, metrics_bar.png")


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
