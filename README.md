# Evaluation of AI-Generated Text Detection Methods
CSCI 544 course project on evaluating the robustness of AI-generated text detection methods, such as DetectGPT and watermark-based detection models.


# Dataset Preparation Guide
install dependencies using `pip install -r requirements.txt` and run `python build_datasets.py` to prepare the project datasets.

## What this script does
`build_datasets.py` is the dataset preparation pipeline for this project.

It does the following:

1. Loads the RAID dataset
2. Saves a local cached copy of the RAID split in `data/` so it does not need to be downloaded again later
3. Builds these processed subsets inside data/processed folder:
   - `plain_human`
   - `plain_ai`
   - `paraphrased_ai`

