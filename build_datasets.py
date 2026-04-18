#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, List

import pandas as pd


FINAL_COLUMNS = [
    "id",
    "text",
    "label",
    "variant",
    "source_dataset",
    "generator_model",
    "attack_type",
    "domain",
    "group_id",
    "split",
    "metadata",
]


# -----------------------------
# basic utilities
# -----------------------------
def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def clean_text(text: object) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def normalize_nullable_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"", "null", "none", "nan"}:
        return None
    return s


def normalize_lower_string(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"", "null", "none", "nan"}:
        return None
    return s


def maybe_sample(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    info(f"Sampling {max_rows:,} rows from {len(df):,} rows")
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def ensure_output_dirs(base_dir: str = "data") -> tuple[Path, Path]:
    base = Path(base_dir)
    processed = base / "processed"
    base.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    info(f"Using base data directory: {base.resolve()}")
    info(f"Using processed directory: {processed.resolve()}")
    return base, processed


# -----------------------------
# RAID load + processing
# -----------------------------
def load_raid_dataframe(split: str = "train", base_dir: str = "data") -> pd.DataFrame:
    """
    Prefer local cached CSV under data/<split>.csv.
    If missing, load with raid-bench and save local copy.
    Fallback to Hugging Face datasets if needed.
    """
    base_path = Path(base_dir)
    local_csv = base_path / f"{split}.csv"

    if local_csv.exists():
        info(f"Found local cached RAID file: {local_csv}")
        info("Loading RAID from local CSV...")
        df = pd.read_csv(local_csv, chunksize=100_000)

        df = pd.concat(
            chunk.sample(frac=0.01, random_state=42)
            for chunk in df
        )
        info(f"Loaded local RAID CSV with {len(df):,} rows")
        return df

    try:
        from raid.utils import load_data  # type: ignore

        info("Local RAID CSV not found.")
        info("Loading RAID with raid-bench (this may download a large file)...")
        df = load_data(split=split) # type: ignore
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        info(f"RAID loaded with {len(df):,} rows")
        info(f"Saving local cache to {local_csv} ...")
        df.to_csv(local_csv, index=False)
        info("Local RAID cache saved successfully")
        return df

    except Exception as e1:
        warn(f"raid-bench load failed: {e1}")

    try:
        from datasets import load_dataset  # type: ignore

        info("Trying Hugging Face datasets fallback...")
        ds = load_dataset("liamdugan/raid", split=split)
        df = ds.to_pandas()
        info(f"RAID loaded from Hugging Face with {len(df):,} rows")
        info(f"Saving local cache to {local_csv} ...")
        df.to_csv(local_csv, index=False)
        info("Local RAID cache saved successfully")
        return df
    except Exception as e2:
        raise RuntimeError(
            "Failed to load RAID with both raid-bench and datasets.\n"
            f"raid-bench error: {e1}\n"
            f"datasets error: {e2}"
        )


def validate_raid_columns(df: pd.DataFrame) -> None:
    required = ["id", "generation", "attack", "domain"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RAID missing required columns: {missing}")

    optional_defaults = {
        "model": None,
        "prompt": None,
        "source_id": None,
        "adv_source_id": None,
        "decoding": None,
        "repetition_penalty": None,
        "label": None,
    }
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in ["prompt", "source_id", "adv_source_id"]:
        df[col] = df[col].apply(normalize_nullable_value)

    df["model"] = df["model"].apply(normalize_nullable_value)
    df["model"] = df["model"].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df["label"] = df["label"].apply(normalize_lower_string)
    df["attack"] = df["attack"].apply(lambda x: "" if pd.isna(x) else str(x).strip().lower())
    df["domain"] = df["domain"].apply(lambda x: "" if pd.isna(x) else str(x).strip())

    info("RAID column validation complete")


def print_debug_value_counts(df: pd.DataFrame, debug: bool = False) -> None:
    if not debug:
        return

    info("Debugging RAID label / attack / model values...")
    print("\n[label value counts]")
    print(df["label"].astype(str).value_counts(dropna=False).head(20))
    print("\n[attack value counts]")
    print(df["attack"].astype(str).value_counts(dropna=False).head(20))
    print("\n[model value counts]")
    print(df["model"].astype(str).value_counts(dropna=False).head(20))


def build_group_id(row: pd.Series) -> str:
    adv = normalize_nullable_value(row.get("adv_source_id"))
    src = normalize_nullable_value(row.get("source_id"))
    if adv is not None:
        return adv
    if src is not None:
        return src
    return ""


def build_metadata_from_raid(row: pd.Series) -> str:
    md = {
        "prompt": row.get("prompt"),
        "source_id": row.get("source_id"),
        "adv_source_id": row.get("adv_source_id"),
        "decoding": row.get("decoding"),
        "repetition_penalty": row.get("repetition_penalty"),
        "raid_attack_raw": row.get("attack"),
        "raid_model_raw": row.get("model"),
        "raid_label_raw": row.get("label"),
    }
    md = {k: v for k, v in md.items() if pd.notna(v) and v not in ["", None]}
    return json.dumps(md, ensure_ascii=False)


def standardize_raid_subset(
    df: pd.DataFrame,
    *,
    label: str,
    variant: str,
    split: str,
) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = df["id"].astype(str)
    out["text"] = df["generation"].apply(clean_text)
    out["label"] = label
    out["variant"] = variant
    out["source_dataset"] = "RAID"
    out["generator_model"] = df["model"].apply(lambda x: "" if x is None else str(x))
    out["attack_type"] = df["attack"].apply(lambda x: "" if x in {"", "none"} else str(x))
    out["domain"] = df["domain"].fillna("")
    out["group_id"] = df.apply(build_group_id, axis=1)
    out["split"] = split
    out["metadata"] = df.apply(build_metadata_from_raid, axis=1)
    out = out[FINAL_COLUMNS].copy()
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out


def get_plain_human(raid_df: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = raid_df[(raid_df["model"] == "human") & (raid_df["attack"] == "none")].copy()
    info(f"plain_human raw rows found: {len(subset):,}")
    return standardize_raid_subset(subset, label="human", variant="plain_human", split=split)


def get_plain_ai(raid_df: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = raid_df[(raid_df["model"] != "human") & (raid_df["attack"] == "none")].copy()
    info(f"plain_ai raw rows found: {len(subset):,}")
    return standardize_raid_subset(subset, label="ai", variant="plain_ai", split=split)


def get_paraphrased_ai(raid_df: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = raid_df[(raid_df["model"] != "human") & (raid_df["attack"] == "paraphrase")].copy()
    info(f"paraphrased_ai raw rows found: {len(subset):,}")
    return standardize_raid_subset(subset, label="ai", variant="paraphrased_ai", split=split)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info(f"Saving CSV to {path} ...")
    df.to_csv(path, index=False)
    info(f"Saved {len(df):,} rows to {path}")


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info(f"Saving JSONL to {path} ...")
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    info(f"Saved {len(df):,} rows to {path}")


def process_raid(split: str, base_dir: str, seed: int, max_rows_per_raid_subset: Optional[int], debug: bool) -> None:
    base_dir_path, processed_dir = ensure_output_dirs(base_dir)
    info("Starting RAID load step...")
    raid_df = load_raid_dataframe(split=split, base_dir=str(base_dir_path))
    validate_raid_columns(raid_df)
    print_debug_value_counts(raid_df, debug=debug)
    info(f"Total RAID rows loaded: {len(raid_df):,}")

    plain_human_df = maybe_sample(get_plain_human(raid_df, split), max_rows_per_raid_subset, seed)
    plain_ai_df = maybe_sample(get_plain_ai(raid_df, split), max_rows_per_raid_subset, seed)
    paraphrased_ai_df = maybe_sample(get_paraphrased_ai(raid_df, split), max_rows_per_raid_subset, seed)

    save_csv(plain_human_df, processed_dir / "plain_human.csv")
    save_csv(plain_ai_df, processed_dir / "plain_ai.csv")
    save_csv(paraphrased_ai_df, processed_dir / "paraphrased_ai.csv")

    all_df = pd.concat([plain_human_df, plain_ai_df, paraphrased_ai_df], ignore_index=True)
    save_csv(all_df, processed_dir / "all_raid.csv")
    save_jsonl(all_df, processed_dir / "all_raid.jsonl")
    info("RAID processing complete")


# -----------------------------
# MarkLLM watermark generation
# -----------------------------
def build_markllm_generator(
    algorithm_name: str = "KGW",
    model_name: str = "facebook/opt-125m",
    max_new_tokens: int = 160,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from markllm.watermark.auto_watermark import AutoWatermark
    from markllm.utils.transformers_config import TransformersConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Loading watermark model {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()

    tf_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

    watermark = AutoWatermark.load(
        algorithm_name,
        algorithm_config="kgw_config.json",
        transformers_config=tf_config,
    )
    info("MarkLLM watermark generator loaded successfully")
    return watermark


def build_generation_prompt(raw_prompt: str, prompt_mode: str) -> str:
    prompt = clean_text(raw_prompt)
    if prompt_mode == "raw":
        return prompt
    if prompt_mode == "chat":
        return (
            "You are a helpful assistant. Answer the following prompt with a standalone response. "
            "Do not repeat the prompt text.\n\n"
            f"Prompt: {prompt}\n\n"
            "Response:"
        )
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


COPIED_PREFIX_MARKERS = [
    "prompt:",
    "response:",
    "answer:",
]


def strip_prompt_echo(full_text: str, used_prompt: str) -> str:
    text = clean_text(full_text)
    prompt = clean_text(used_prompt)
    if not text:
        return ""

    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()

    lowered = text.lower()
    for marker in COPIED_PREFIX_MARKERS:
        if lowered.startswith(marker):
            text = text[len(marker):].strip()
            lowered = text.lower()

    return clean_text(text)


MIN_ACCEPTABLE_WORDS = 8


def looks_bad_generation(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_ACCEPTABLE_WORDS:
        return True
    if text.lower().startswith("prompt:"):
        return True
    return False


def standardize_watermarked_rows(
    source_rows: pd.DataFrame,
    generated_texts: List[str],
    *,
    generator_model_name: str,
    split: str,
    watermark_algorithm: str,
    prompt_mode: str,
) -> pd.DataFrame:
    if len(source_rows) != len(generated_texts):
        raise ValueError("source_rows and generated_texts must have the same length")

    out = pd.DataFrame()
    out["id"] = source_rows["id"].astype(str) + "_wm"
    out["text"] = [clean_text(x) for x in generated_texts]
    out["label"] = "ai"
    out["variant"] = "watermarked_ai"
    out["source_dataset"] = "RAID+MarkLLM"
    out["generator_model"] = generator_model_name
    out["attack_type"] = ""
    out["domain"] = source_rows["domain"].fillna("")
    out["group_id"] = source_rows["group_id"].fillna("")
    out["split"] = split

    md_list = []
    for _, row in source_rows.iterrows():
        md_list.append(json.dumps({
            "parent_id": row["id"],
            "parent_variant": row["variant"],
            "watermark_source_dataset": "MarkLLM",
            "watermark_algorithm": watermark_algorithm,
            "prompt_mode": prompt_mode,
        }, ensure_ascii=False))
    out["metadata"] = md_list

    out = out[FINAL_COLUMNS].copy()
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out


def generate_watermarked_ai(
    raid_df_full: pd.DataFrame,
    watermark,
    *,
    split: str,
    watermark_model_name: str,
    watermark_algorithm: str,
    prompt_mode: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    info("Preparing RAID rows for watermark generation...")

    source_df = raid_df_full[(raid_df_full["prompt"].notna()) & (raid_df_full["prompt"].astype(str).str.strip() != "")].copy()

    # Prefer non-human rows with attack=none to match your AI generation setting.
    preferred = source_df[(source_df["model"] != "human") & (source_df["attack"] == "none")].copy()
    if len(preferred) >= max_rows:
        source_df = preferred
    else:
        source_df = preferred if len(preferred) > 0 else source_df

    source_df = maybe_sample(source_df, max_rows=max_rows, seed=seed).reset_index(drop=True)
    info(f"Rows selected for watermark generation: {len(source_df):,}")

    standardized_source = standardize_raid_subset(
        source_df,
        label="ai",
        variant="plain_ai_prompt_source",
        split=split,
    )

    prompt_map = {str(row["id"]): str(row["prompt"]).strip() for _, row in source_df.iterrows()}

    generated_texts: List[str] = []
    total = len(standardized_source)
    for i, (_, row) in enumerate(standardized_source.iterrows(), start=1):
        row_id = str(row["id"])
        raw_prompt = prompt_map[row_id]
        used_prompt = build_generation_prompt(raw_prompt, prompt_mode=prompt_mode)

        full_output = watermark.generate_watermarked_text(used_prompt)
        cleaned_output = strip_prompt_echo(full_output, used_prompt)

        if looks_bad_generation(cleaned_output) and prompt_mode == "chat":
            fallback_output = watermark.generate_watermarked_text(raw_prompt)
            cleaned_output = strip_prompt_echo(fallback_output, raw_prompt)

        generated_texts.append(cleaned_output)

        if i == 1 or i % 10 == 0 or i == total:
            info(f"Watermarked generation progress: {i:,}/{total:,}")

    result = standardize_watermarked_rows(
        standardized_source,
        generated_texts,
        generator_model_name=watermark_model_name,
        split=split,
        watermark_algorithm=watermark_algorithm,
        prompt_mode=prompt_mode,
    )

    return result


# -----------------------------
# command handlers
# -----------------------------
def cmd_download_raid(args) -> None:
    base_dir, _ = ensure_output_dirs(args.base_dir)
    df = load_raid_dataframe(split=args.split, base_dir=str(base_dir))
    validate_raid_columns(df)
    info(f"RAID download/cache ready: {len(df):,} rows")


def cmd_process_raid(args) -> None:
    process_raid(
        split=args.split,
        base_dir=args.base_dir,
        seed=args.seed,
        max_rows_per_raid_subset=args.max_rows_per_raid_subset,
        debug=args.debug,
    )


def cmd_watermark(args) -> None:
    base_dir, processed_dir = ensure_output_dirs(args.base_dir)

    raid_df = load_raid_dataframe(split=args.split, base_dir=str(base_dir))
    validate_raid_columns(raid_df)

    watermark = build_markllm_generator(
        algorithm_name=args.watermark_algorithm,
        model_name=args.watermark_model,
        max_new_tokens=args.max_new_tokens,
    )

    watermark_df = generate_watermarked_ai(
        raid_df,
        watermark,
        split=args.split,
        watermark_model_name=args.watermark_model,
        watermark_algorithm=args.watermark_algorithm,
        prompt_mode=args.prompt_mode,
        max_rows=args.max_watermark_rows,
        seed=args.seed,
    )

    out_name = f"watermarked_ai_{args.max_watermark_rows}.csv"
    save_csv(watermark_df, processed_dir / out_name)
    save_jsonl(watermark_df, processed_dir / out_name.replace(".csv", ".jsonl"))
    info("Watermark generation complete")


def cmd_all(args) -> None:
    process_raid(
        split=args.split,
        base_dir=args.base_dir,
        seed=args.seed,
        max_rows_per_raid_subset=args.max_rows_per_raid_subset,
        debug=args.debug,
    )
    cmd_watermark(args)


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RAID subsets and generate MarkLLM watermarked text.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split", type=str, default="train")
        p.add_argument("--base-dir", type=str, default="data")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--debug", action="store_true")

    p_download = subparsers.add_parser("download_raid", help="Download/cache RAID only.")
    add_common_arguments(p_download)
    p_download.set_defaults(func=cmd_download_raid)

    p_process = subparsers.add_parser("process_raid", help="Build plain_human/plain_ai/paraphrased_ai from cached RAID.")
    add_common_arguments(p_process)
    p_process.add_argument("--max-rows-per-raid-subset", type=int, default=None)
    p_process.set_defaults(func=cmd_process_raid)

    p_watermark = subparsers.add_parser("watermark", help="Generate watermarked text from cached RAID prompts.")
    add_common_arguments(p_watermark)
    p_watermark.add_argument("--max-watermark-rows", type=int, default=100)
    p_watermark.add_argument("--watermark-algorithm", type=str, default="KGW")
    p_watermark.add_argument("--watermark-model", type=str, default="facebook/opt-125m")
    p_watermark.add_argument("--max-new-tokens", type=int, default=160)
    p_watermark.add_argument("--prompt-mode", choices=["raw", "chat"], default="chat")
    p_watermark.set_defaults(func=cmd_watermark)

    p_all = subparsers.add_parser("all", help="Process RAID and then generate watermarked text.")
    add_common_arguments(p_all)
    p_all.add_argument("--max-rows-per-raid-subset", type=int, default=None)
    p_all.add_argument("--max-watermark-rows", type=int, default=100)
    p_all.add_argument("--watermark-algorithm", type=str, default="KGW")
    p_all.add_argument("--watermark-model", type=str, default="facebook/opt-125m")
    p_all.add_argument("--max-new-tokens", type=int, default=160)
    p_all.add_argument("--prompt-mode", choices=["raw", "chat"], default="chat")
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
