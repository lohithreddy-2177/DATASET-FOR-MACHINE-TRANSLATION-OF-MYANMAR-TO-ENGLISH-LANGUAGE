# BENCHMARK-FOR-MACHINE-TRANSLATOIN-OF-MYANMAR-TO-ENGLISH-LANGUAGE

This repository contains a custom Myanmar (Burmese) → English Benchmark (parallel dataset) and a collection of pretrained models fine-tuned on it for machine translation. The repository also includes full evaluation scripts and results across multiple standard MT metrics.

SUMMARY

Task: Myanmar (my) → English (en) Neural Machine Translation
Approach: Fine-tuning existing pretrained multilingual/translation models on a custom Myanmar–English parallel corpus
Evaluation: BLEU, chrF, ROUGE-2, ROUGE-4, BERTScore, BLEURT

REPO STRUCTURE
~~~
├── data/
│   ├── processed/         # Cleaned & tokenized train/val/test splits
|       |-train.txt
        |-test.txt 
│   └── README.md           # Dataset-specific documentation
├── models/                 # Fine-tuned models checkpoints
│   ├── model_1/             
│   ├── model_2/             
│   └── ...
├── scripts/
│   ├── preprocess.py        # Data cleaning & tokenization
│   ├── train.py              # Fine-tuning script
│   ├── translate.py          # Inference / translation script
│   └── evaluate.py           # Computes BLEU, chrF, ROUGE, BERTScore, BLEURT
|   ├──Finetuned scripts      # finetuned scripts of pretrained models
├── results/
│   └── metrics_summary.csv   # Consolidated evaluation results
├── requirements.txt
└── README.md
~~~

DATASET DESCRIPTION

Language pair: Burmese (my) ↔ English (en)  
Size: <train_size> sentence pairs (train) / <val_size> (validation) / <test_size> (test)  
Domain(s): <e.g., news, religious texts, general web text, government documents>  
Sources: <>  
Preprocessing: <tokenization, normalization, deduplication, filtering by length/ratio, etc.>  

Dataset files are available under data/ in .txt format with source (Burmese) and target (English) fields  

Fine-Tuned Models

The following pretrained models were fine-tuned on the dataset described above:

|          Model               | Trainable Parameters |        Targeted modules       |    Training Key-attributes       |      Notes    |
|------------------------------|----------------------|-------------------------------|----------------------------------|---------------|
| `<Gemma-SEA-LION-v3-9B-IT> ` |                      | `<size>`                      | `<n>`                            | `<notes>`     |
| ` nllb-200-3.3B `            |                      | `<size>`                      | `<n>`                            | `<notes>`     |
| `<model_3_name> `            |                      | `<size>`                      | `<n>`                            | `<notes>`     |



