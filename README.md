# Evaluation of AI-Generated Text Detection Methods
CSCI 544 course project on evaluating the robustness of AI-generated text detection methods, such as DetectGPT and watermark-based detection models.


# Setup

Install dependencies:

```bash
pip install -r requirements.txt
```


# Dataset Preparation

`build_datasets.py` is the dataset preparation pipeline. It downloads the RAID dataset, processes it into subsets, and generates watermarked text using the GPT-2 model.

Run all steps at once:

```bash
python build_datasets.py all \
    --watermark-model gpt2 \
    --max-watermark-rows 100
```

Or run each step individually:

```bash
# Step 1: Download and cache the RAID dataset
python build_datasets.py download_raid

# Step 2: Build plain_human, plain_ai, and paraphrased_ai subsets
python build_datasets.py process_raid

# Step 3: Generate watermarked text using GPT-2
python build_datasets.py watermark \
    --watermark-model gpt2 \
    --max-watermark-rows 100
```

This produces the following files in `data/processed/`:

| File | Description |
|------|-------------|
| `plain_human.csv` | Human-written text |
| `plain_ai.csv` | AI-generated text (no attack) |
| `paraphrased_ai.csv` | AI-generated text after paraphrasing attack |
| `watermarked_ai_100.csv` | AI-generated text with KGW watermark (GPT-2) |

Each file uses a unified schema with fields: `id`, `text`, `label`, `variant`, `source_dataset`, `generator_model`, `attack_type`, `domain`, `group_id`, `split`, `metadata`.


# Running Experiments

## DetectGPT

`run_detectgpt_on_datasets.py` runs DetectGPT using GPT-2 as the base model and `t5-large` as the mask model. Results are saved under `eval_results/<experiment_name>/results.json`.

**Experiment 1 — Human vs. Plain AI:**

```bash
python run_detectgpt_on_datasets.py \
    --human_csv data/processed/plain_human.csv \
    --ai_csv data/processed/plain_ai.csv \
    --experiment_name human_vs_plain_ai \
    --base_model_name gpt2 \
    --generator_model gpt2 \
    --n_samples 100 \
    --n_perturbations 50
```

**Experiment 2 — Human vs. Paraphrased AI:**

```bash
python run_detectgpt_on_datasets.py \
    --human_csv data/processed/plain_human.csv \
    --ai_csv data/processed/paraphrased_ai.csv \
    --experiment_name human_vs_paraphrased \
    --base_model_name gpt2 \
    --generator_model gpt2 \
    --n_samples 100 \
    --n_perturbations 50
```

**Experiment 3 — Human vs. Watermarked AI:**

```bash
python run_detectgpt_on_datasets.py \
    --human_csv data/processed/plain_human.csv \
    --ai_csv data/processed/watermarked_ai_100.csv \
    --experiment_name human_vs_watermarked \
    --base_model_name gpt2 \
    --generator_model gpt2 \
    --n_samples 100 \
    --n_perturbations 50
```

## KGW Watermark Detection

`evaluate_watermark.py` scores text using the KGW watermark detector (GPT-2, configured via `kgw_config.json`) and reports AUROC and FPR@95TPR. Results are saved to the path given by `--output`.

**Experiment 1 — Human vs. Watermarked AI:**

```bash
python evaluate_watermark.py \
    --dataset1 data/processed/plain_human.csv \
    --dataset2 data/processed/watermarked_ai_100.csv \
    --n_samples 100 \
    --output eval_results/watermark_human_vs_watermarked.json
```

**Experiment 2 — Plain AI vs. Watermarked AI:**

```bash
python evaluate_watermark.py \
    --dataset1 data/processed/plain_ai.csv \
    --dataset2 data/processed/watermarked_ai_100.csv \
    --n_samples 100 \
    --output eval_results/watermark_plain_ai_vs_watermarked.json
```

**Experiment 3 — Human vs. Plain AI:**

```bash
python evaluate_watermark.py \
    --dataset1 data/processed/plain_human.csv \
    --dataset2 data/processed/plain_ai.csv \
    --n_samples 100 \
    --output eval_results/watermark_human_vs_plain_ai.json
```
