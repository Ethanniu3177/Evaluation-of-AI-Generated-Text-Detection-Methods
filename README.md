# Evaluation of AI-Generated Text Detection Methods

CSCI 544 course project evaluating the robustness of AI-generated text detection methods, including DetectGPT and KGW watermark-based detection.

---

## Environment Setup

**Requirements:** Python 3.9+, pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include `torch`, `transformers`, `datasets`, `markllm`, `pandas`, `scikit-learn`, `accelerate`, and `sentencepiece`.

---

## System

Experiments were run on **Google Colab** with a GPU runtime (NVIDIA T4 or equivalent). The code automatically selects CUDA if available, falling back to CPU:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

GPU is strongly recommended — DetectGPT with 50 perturbations per sample is computationally intensive. On CPU, runtimes will be significantly longer.

---

## Running the Code

### Step 1 — Prepare Datasets

`build_datasets.py` downloads the RAID dataset, builds text subsets, and generates watermarked text using GPT-2.

Run all steps at once:

```bash
python build_datasets.py all \
    --watermark-model gpt2 \
    --max-watermark-rows 100
```

Or run each step individually:

```bash
# Download and cache the RAID dataset
python build_datasets.py download_raid

# Build plain_human, plain_ai, and paraphrased_ai subsets
python build_datasets.py process_raid

# Generate watermarked text using GPT-2
python build_datasets.py watermark \
    --watermark-model gpt2 \
    --max-watermark-rows 100
```

Output files in `data/processed/`:

| File | Description |
|------|-------------|
| `plain_human.csv` | Human-written text |
| `plain_ai.csv` | AI-generated text (no attack) |
| `paraphrased_ai.csv` | AI-generated text after paraphrasing attack |
| `watermarked_ai_100.csv` | AI-generated text with KGW watermark (GPT-2) |

Each file uses a unified schema: `id`, `text`, `label`, `variant`, `source_dataset`, `generator_model`, `attack_type`, `domain`, `group_id`, `split`, `metadata`.

---

### Step 2 — Run DetectGPT Experiments

`run_detectgpt_on_datasets.py` runs DetectGPT using GPT-2 as the base model and `t5-large` as the mask model.

**Baseline 1 — Human vs. Plain AI:**

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

**Baseline 2 — Human vs. Watermarked AI:**

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

**Paraphrasing 1 — Human vs. Paraphrased AI:**

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

Results are saved to `eval_results/<experiment_name>/results.json`.

---

### Step 3 — Run KGW Watermark Detection Experiments

`evaluate_watermark.py` scores text using the KGW watermark detector (GPT-2, configured via `kgw_config.json`) and reports AUROC and FPR@95TPR.

**Baseline 3 — Plain AI vs. Watermarked AI:**

```bash
python evaluate_watermark.py \
    --dataset1 data/processed/plain_ai.csv \
    --dataset2 data/processed/watermarked_ai_100.csv \
    --n_samples 100 \
    --output eval_results/watermark_plain_ai_vs_watermarked.json
```

**Baseline 4 — Human vs. Watermarked AI:**

```bash
python evaluate_watermark.py \
    --dataset1 data/processed/plain_human.csv \
    --dataset2 data/processed/watermarked_ai_100.csv \
    --n_samples 100 \
    --output eval_results/watermark_human_vs_watermarked.json
```

**Baseline 5 — Human vs. Plain AI:**

```bash
python evaluate_watermark.py \
    --dataset1 data/processed/plain_human.csv \
    --dataset2 data/processed/plain_ai.csv \
    --n_samples 100 \
    --output eval_results/watermark_human_vs_plain_ai.json
```

---

### Step 4 — Generate Plots

`generate_plots.py` reads all experiment result files from `experiment_results/` and produces the following plots in `plots/`:

- ROC curves
- AUROC bar chart
- Score distribution histograms
- FPR@95TPR bar chart

```bash
python generate_plots.py
```

---

## How Results Are Generated

**DetectGPT** works by perturbing a text sample `n_perturbations` times using a mask-filling model (`t5-large`), then comparing the log-likelihood of the original text against its perturbations under a base language model (GPT-2). AI-generated text tends to occupy local maxima in the model's probability landscape, so a higher perturbation discrepancy score indicates AI authorship.

**KGW Watermark Detection** scores text by checking whether output tokens fall in a statistically elevated "green list" partition, which is determined at generation time using a secret hash key. The detector computes a z-score against the null hypothesis of no watermark. Configuration is set in `kgw_config.json` (gamma=0.5, delta=2.0, z-threshold=4.0).

Both detectors report **AUROC** (area under the ROC curve) and **FPR@95TPR** (false positive rate at 95% true positive rate) as evaluation metrics.
