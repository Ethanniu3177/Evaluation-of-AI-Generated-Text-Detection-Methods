# Evaluation of AI-Generated Text Detection Methods
CSCI 544 course project on evaluating the robustness of AI-generated text detection methods, such as DetectGPT and watermark-based detection models.


# Dataset Preparation Guide
install dependencies using `pip install -r requirements.txt` and run `python build_datasets.py` to prepare the project datasets.

`build_datasets.py` is the dataset preparation pipeline for this project.

The processed RAID-based datasets use a unified schema:
- id: unique sample id
- text: text content used for testing
- label: human or ai
- variant: plain_human, plain_ai, paraphrased_ai, or watermarked_ai
- source_dataset: source of the sample, such as RAID or RAID+MarkLLM
- generator_model: model name if available
- attack_type: attack type if applicable, such as paraphrase
- domain: RAID domain field
- group_id: linkage id derived from RAID source ids when available
- split: dataset split, such as train
- metadata: extra JSON-style source information

It does the following:
1. Loads the RAID dataset
2. Saves a local cached copy of the RAID split in `data/` so it does not need to be downloaded again later
3. Builds these processed subsets inside data/processed folder:
   - `plain_human`
   - `plain_ai`
   - `paraphrased_ai`

