#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ensure_output_dirs(base_dir: str = "data") -> tuple[Path, Path]:
    base = Path(base_dir)
    processed = base / "processed"
    base.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    info(f"Using base data directory: {base.resolve()}")
    info(f"Using processed directory: {processed.resolve()}")
    return base, processed


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
        df = pd.read_csv(local_csv)
        info(f"Loaded local RAID CSV with {len(df):,} rows")
        return df

    try:
        from raid.utils import load_data  # type: ignore

        info("Local RAID CSV not found.")
        info("Loading RAID with raid-bench (this may download a large file)...")
        df = load_data(split=split)
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


def clean_text(text: object) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


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
    info(f"Columns available: {list(df.columns)}")


def print_debug_value_counts(df: pd.DataFrame, debug: bool = False) -> None:
    if not debug:
        return

    info("Debugging RAID label / attack / model values...")

    print("\n[label value counts]")
    print(df["label"].astype(str).value_counts(dropna=False).head(20))

    print("\n[attack value counts]")
    print(df["attack"].astype(str).value_counts(dropna=False).head(20))

    print("\n[model null count]")
    print(df["model"].isna().sum())

    print("\n[model sample values]")
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

    if variant == "paraphrased_ai":
        out["attack_type"] = df["attack"].apply(lambda x: "" if x in {"", "none"} else str(x))
    else:
        out["attack_type"] = ""

    out["domain"] = df["domain"].fillna("")
    out["group_id"] = df.apply(build_group_id, axis=1)
    out["split"] = split
    out["metadata"] = df.apply(build_metadata_from_raid, axis=1)

    out = out[FINAL_COLUMNS].copy()
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out


def get_plain_human(raid_df: pd.DataFrame, split: str) -> pd.DataFrame:
    info("Building plain_human subset...")

    subset = raid_df[
        (raid_df["model"] == "human") &
        (raid_df["attack"] == "none")
    ].copy()

    info(f"plain_human raw rows found: {len(subset):,}")
    return standardize_raid_subset(subset, label="human", variant="plain_human", split=split)


def get_plain_ai(raid_df: pd.DataFrame, split: str) -> pd.DataFrame:
    info("Building plain_ai subset...")

    subset = raid_df[
        (raid_df["model"] != "human") &
        (raid_df["attack"] == "none")
    ].copy()

    info(f"plain_ai raw rows found: {len(subset):,}")
    return standardize_raid_subset(subset, label="ai", variant="plain_ai", split=split)


def get_paraphrased_ai(raid_df: pd.DataFrame, split: str) -> pd.DataFrame:
    info("Building paraphrased_ai subset...")

    subset = raid_df[
        (raid_df["model"] != "human") &
        (raid_df["attack"] == "paraphrase")
    ].copy()

    info(f"paraphrased_ai raw rows found: {len(subset):,}")
    return standardize_raid_subset(subset, label="ai", variant="paraphrased_ai", split=split)


def maybe_sample(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    info(f"Sampling {max_rows:,} rows from {len(df):,} rows")
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def build_markllm_generator(
    algorithm_name: str = "KGW",
    model_name: str = "facebook/opt-1.3b",
    max_new_tokens: int = 200,
):
    """
    Build MarkLLM watermark generator.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from markllm.watermark.auto_watermark import AutoWatermark
    from markllm.utils.transformers_config import TransformersConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Loading watermark model {model_name} on {device}...")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        transformers_config=tf_config,
    )
    info("MarkLLM watermark generator loaded successfully")
    return watermark


def standardize_watermarked_rows(
    source_rows: pd.DataFrame,
    generated_texts: List[str],
    *,
    generator_model_name: str,
    split: str,
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
        }, ensure_ascii=False))
    out["metadata"] = md_list

    out = out[FINAL_COLUMNS].copy()
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out


def generate_watermarked_ai(
    standardized_df: pd.DataFrame,
    raid_df_full: pd.DataFrame,
    watermark,
    *,
    split: str,
    watermark_model_name: str,
    max_rows: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """
    Generate watermarked AI from RAID prompts.
    We match standardized rows back to RAID by id, then use RAID prompt.
    """
    info("Preparing rows for watermark generation...")

    raid_lookup = raid_df_full.set_index(raid_df_full["id"].astype(str), drop=False)

    rows_for_generation = []

    for _, row in standardized_df.iterrows():
        rid = str(row["id"])
        if rid not in raid_lookup.index:
            continue
        raid_row = raid_lookup.loc[rid]
        prompt = raid_row.get("prompt")
        if pd.isna(prompt) or str(prompt).strip() == "":
            continue
        rows_for_generation.append(row)

    if not rows_for_generation:
        warn("No rows with non-empty prompts were found for watermark generation")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    source_rows = pd.DataFrame(rows_for_generation)
    info(f"Rows with usable prompts before sampling: {len(source_rows):,}")
    source_rows = maybe_sample(source_rows, max_rows=max_rows, seed=seed).reset_index(drop=True)
    info(f"Rows selected for watermark generation: {len(source_rows):,}")

    sampled_prompts = []
    for _, row in source_rows.iterrows():
        raid_row = raid_lookup.loc[str(row["id"])]
        sampled_prompts.append(str(raid_row["prompt"]).strip())

    generated_texts = []
    total = len(sampled_prompts)
    info(f"Generating watermarked_ai for {total:,} rows...")

    for i, prompt in enumerate(sampled_prompts, start=1):
        wm_text = watermark.generate_watermarked_text(prompt)
        generated_texts.append(wm_text)

        if i == 1 or i % 10 == 0 or i == total:
            info(f"Watermarked generation progress: {i:,}/{total:,}")

    return standardize_watermarked_rows(
        source_rows,
        generated_texts,
        generator_model_name=watermark_model_name,
        split=split,
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--base-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-raid-subset", type=int, default=None)
    # parser.add_argument("--generate-watermarks", action="store_true")
    # parser.add_argument("--max-watermark-rows", type=int, default=100)
    # parser.add_argument("--watermark-algorithm", type=str, default="KGW")
    # parser.add_argument("--watermark-model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    base_dir, processed_dir = ensure_output_dirs(args.base_dir)

    info("Starting RAID load step...")
    raid_df = load_raid_dataframe(split=args.split, base_dir=str(base_dir))
    validate_raid_columns(raid_df)
    print_debug_value_counts(raid_df, debug=args.debug)
    info(f"Total RAID rows loaded: {len(raid_df):,}")

    plain_human_df = maybe_sample(get_plain_human(raid_df, args.split), args.max_rows_per_raid_subset, args.seed)
    plain_ai_df = maybe_sample(get_plain_ai(raid_df, args.split), args.max_rows_per_raid_subset, args.seed)
    paraphrased_ai_df = maybe_sample(get_paraphrased_ai(raid_df, args.split), args.max_rows_per_raid_subset, args.seed)

    save_csv(plain_human_df, processed_dir / "plain_human.csv")
    save_csv(plain_ai_df, processed_dir / "plain_ai.csv")
    save_csv(paraphrased_ai_df, processed_dir / "paraphrased_ai.csv")

    # watermark_df = pd.DataFrame(columns=FINAL_COLUMNS)

    # if args.generate_watermarks:
    #     info("Beginning watermark generation setup...")
    #     watermark = build_markllm_generator(
    #         algorithm_name=args.watermark_algorithm,
    #         model_name=args.watermark_model,
    #         max_new_tokens=args.max_new_tokens,
    #     )

    #     source_for_watermark = pd.concat(
    #         [plain_human_df, plain_ai_df, paraphrased_ai_df],
    #         ignore_index=True
    #     )

    #     watermark_df = generate_watermarked_ai(
    #         source_for_watermark,
    #         raid_df,
    #         watermark,
    #         split=args.split,
    #         watermark_model_name=args.watermark_model,
    #         max_rows=args.max_watermark_rows,
    #         seed=args.seed,
    #     )

    #     save_csv(watermark_df, processed_dir / "watermarked_ai.csv")
    # else:
    #     info("Skipping watermarked_ai generation. Add --generate-watermarks to enable it.")

    # all_df = pd.concat(
    #     [plain_human_df, plain_ai_df, paraphrased_ai_df, watermark_df],
    #     ignore_index=True
    # )

    # save_csv(all_df, processed_dir / "all_data.csv")
    # save_jsonl(all_df, processed_dir / "all_data.jsonl")

    # info("Done.")
    # info(f"Files are saved under: {processed_dir.resolve()}")


if __name__ == "__main__":
    main()