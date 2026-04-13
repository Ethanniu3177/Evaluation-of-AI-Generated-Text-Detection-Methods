import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

tf_config = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    vocab_size=tokenizer.vocab_size,
    device=device,
)

watermark = AutoWatermark.load(
    "KGW", # type: ignore[arg-type]
    algorithm_config="kgw_config.json",  # or dict (recommended)
    transformers_config=tf_config,
)

def detect_watermark(text, seed=42, gamma=0.5):
    """
    Detect KGW watermark in a piece of text.

    Parameters:
        text (str): Input text to check.
        seed (int): Secret RNG seed used during watermarking.
        gamma (float): Expected fraction of green tokens (default 0.5).

    Returns:
        float: Probability score (0-1) of watermark presence.
               Returns 0.5 (neutral) if text is too short to score.
    """
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_tokens = inputs["input_ids"][0].cpu().numpy()

    # Need at least 2 tokens (one prev + one to score)
    if len(input_tokens) < 2:
        return 0.5

    green_count = 0
    total = 0

    # Per-token context-dependent seeding (skip first token — no prior context)
    for i in range(1, len(input_tokens)):
        prev_token = int(input_tokens[i - 1])
        np.random.seed(seed + prev_token)  # mirrors watermark generation

        # Use a set for O(1) lookup
        green_list = set(
            np.random.choice(
                tokenizer.vocab_size,
                size=int(tokenizer.vocab_size * gamma),
                replace=False
            ).tolist()
        )

        if int(input_tokens[i]) in green_list:
            green_count += 1
        total += 1

    expected = total * gamma
    std = np.sqrt(total * gamma * (1 - gamma))

    if std == 0:
        return 0.5

    z = (green_count - expected) / std
    prob = 1 / (1 + np.exp(-z))
    return prob

def detect_with_markllm(text):
    result = watermark.score_text(text)
    return result["z_score"]

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate watermark detection metrics between two datasets"
    )
    parser.add_argument("--dataset1", required=True,
                        help="Path to first CSV (non-watermarked text)")
    parser.add_argument("--dataset2", required=True,
                        help="Path to second CSV (watermarked text)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Max samples per dataset (optional)")
    parser.add_argument("--score_column", default=None,
                        help="Column containing pre-computed detector scores. "
                             "If None, scores are computed by detect_watermark().")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for binary classification")
    parser.add_argument("--output", default="eval_results.json",
                        help="Output JSON file for metrics")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    df1 = pd.read_csv(args.dataset1)
    df2 = pd.read_csv(args.dataset2)

    # Validate required column exists in both dataframes
    if args.score_column is None:
        for name, df in [("dataset1", df1), ("dataset2", df2)]:
            if "text" not in df.columns:
                raise ValueError(f"'{name}' has no 'text' column. "
                                 f"Columns found: {list(df.columns)}")

    if args.n_samples:
        df1 = df1.sample(n=min(args.n_samples, len(df1)),
                         random_state=42).reset_index(drop=True)
        df2 = df2.sample(n=min(args.n_samples, len(df2)),
                         random_state=42).reset_index(drop=True)

    # Assign ground-truth labels before concat so ordering is explicit
    df1 = df1.copy()
    df2 = df2.copy()
    df1["y_true"] = 0  # non-watermarked
    df2["y_true"] = 1  # watermarked

    # Shuffle first — avoids order-dependent metric artifacts
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Compute / retrieve watermark scores AFTER shuffle
    # ------------------------------------------------------------------
    if args.score_column:
        if args.score_column not in df.columns:
            raise ValueError(f"score_column '{args.score_column}' not found. "
                             f"Columns: {list(df.columns)}")
        y_score = df[args.score_column].to_numpy(dtype=float)
    else:
        # Handle missing text rows
        missing = df["text"].isna().sum()
        if missing:
            print(f"Warning: {missing} rows with missing text — filling with empty string.")
            df["text"] = df["text"].fillna("")

        print("Running watermark detector...")

        y_score = np.array(
            [detect_with_markllm(t) for t in df["text"]],
            dtype=float
        )

    y_true = df["y_true"].to_numpy(dtype=int)

    # Guard: AUROC requires both classes present
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true contains only one class — AUROC is undefined. "
                         "Check that both datasets loaded correctly.")

    print(f"y_score sample: {y_score[:5]}")
    print(f"y_true sample:  {y_true[:5]}")
    print(f"Score range: {y_score.min():.4f} - {y_score.max():.4f}")

    y_pred = (y_score >= args.threshold).astype(int)

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    results = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auroc":     round(float(roc_auc_score(y_true, y_score)), 4),
        "n_samples_dataset1": len(df1),
        "n_samples_dataset2": len(df2),
    }

    # ------------------------------------------------------------------
    # Print and save
    # ------------------------------------------------------------------
    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()