import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig

config_filename = "kgw_config.json"
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device) # type: ignore
model.eval()

tf_config = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    vocab_size=tokenizer.vocab_size,
    device=device,
)

watermark = AutoWatermark.load(
    "KGW", # type: ignore[arg-type]
    algorithm_config=config_filename,
    transformers_config=tf_config,
)

def detect_with_markllm(text):
    result = watermark.detect_watermark(text)
    return result["score"]

def main():
    with open(config_filename) as f:
        kgw_config = json.load(f)

    DEFAULT_THRESHOLD = kgw_config.get("z_threshold", 4.0)

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
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
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

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    auroc = roc_auc_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, 0.95)
    fpr_at_95_tpr = float(fpr[min(idx, len(fpr) - 1)])

    results = {
        "auroc":          round(float(auroc), 4),
        "fpr_at_95_tpr":  round(fpr_at_95_tpr, 4),
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