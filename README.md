# BENCHMARK-FOR-MACHINE-TRANSLATOIN-OF-MYANMAR-TO-ENGLISH-LANGUAGE

This repository contains a custom Myanmar (Burmese) → English Benchmark (parallel dataset) and a collection of pretrained models fine-tuned on it for machine translation. In addition, we built a custom model using mBERT as both the encoder and decoder, designed to maximize translation performance for this language pair. The repository also includes full evaluation scripts and results across multiple standard MT metrics.

SUMMARY

Task: Myanmar (my) → English (en) Neural Machine Translation  
Approach: Fine-tuning existing pretrained multilingual/translation models on a custom Myanmar–English parallel corpus  
Evaluation: BLEU, chrF, ROUGE-2, ROUGE-4, BERTScore, BLEURT  

REPO STRUCTURE
~~~
├── data/
│   ├── processed/         # Cleaned & tokenized train/val/test splits
|       |-train.txt
|       |-test.txt 
│   └── README.md           # Dataset-specific documentation
├── models/                 # Fine-tuned models checkpoints u need these checkpoints for using the finetuned models
│   ├── model_1/
|        ├── adapter_config.json
|        ├── adapter_model.safetensors
|        ├── ...            
│   ├── model_2/             
│   └── ...
|─── mBERT2mBERT           # best performance model
|     ├── model.safetensors
|     ├── special_token_map.json
|     ├── tokenizer_config.json
|     ├── model.py
|     ├── inferenec.py
|     ├── .....
├── scripts/                 # finetuned scripts of pretrained models
|    ├── model1/
|        ├── model.py
|        ├── inferenec.py     #use these to directly use the finetune models for translation
|    ├── model2/
|    ├── ....     
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

FINETUNED MODLES

The following pretrained models were fine-tuned on the dataset described above:

|          Model               | Trainable Parameters |        Targeted modules       |    Training Key-attributes       |      Notes    |
|------------------------------|----------------------|-------------------------------|----------------------------------|---------------|
| `<Gemma-SEA-LION-v3-9B-IT> ` |                      | `<size>`                      | `<n>`                            | `<notes>`     |
| ` nllb-200-3.3B `            |                      | `<size>`                      | `<n>`                            | `<notes>`     |
| `<model_3_name> `            |                      | `<size>`                      | `<n>`                            | `<notes>`     |


RESULTS

|          Model               | BLEU    |      BERTSCROE     |    CHRF   |      ROUGE2   |      ROUGE4   |  BLEURTSCORE  |
|------------------------------|---------|--------------------|-----------|---------------|---------------|---------------|
| `<Gemma-SEA-LION-v3-9B-IT> ` |  22.59  |     0.9176         | 43.0454   |   0.4883      |    0.2684     |  0.6142       |
| ` nllb-200-3.3B `            |  32.76  |     0.9276         | 52.135    |   0.3370      |     0.1579    |               |
| `<model_3_name> `            |         |                    | `<n>`     |               |               |               |


CUSTOM ENCODER-DECODER MODEL(mBERT)

In addition to fine-tuning existing pretrained MT models, we built a custom encoder-decoder architecture using mBERT for both the encoder and decoder, aimed at maximizing translation performance specifically for Myanmar → English. This setup allows the decoder to leverage mBERT's multilingual contextual representations directly, rather than relying on a separate decoder architecture.

|          Model               |    ENCODER    |    DECODER    |    PARAMETERS   |      NOTES        |
|------------------------------|---------------|---------------|-----------------|-------------------|
|     `<mBERT2mBERT model>`    |   `<mBERT>`   |   `<mBERT>`   | 43.0454         |   0.4883          |


INSTALLATION

Since model checkpoints are stored with Git LFS, install LFS first, then clone:
~~~
git lfs install
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
~~~
If you already cloned the repo without LFS set up, pull the actual weight files with:
~~~
git lfs pull
~~~
Key dependencies
~~~
transformers
peft
torch
sacrebleu
rouge-score
bert-score
bleurt
sentencepiece
~~~

USAGE

You can use either the fine-tuned pretrained models (LoRA adapters) or the custom mBERT2mBERT model (best performance) for translation. Pick the option below that matches what you want to run.

Using the fine-tuned pretrained model (adapter-based)

Each fine-tuned model has its own folder under scripts/ with a ready-to-use inference.py. The corresponding adapter checkpoint lives in models/<model_name>/.
~~~
python scripts/model1/inference.py \
    --base_model <pretrained-model-name-or-path> \
    --adapter_path models/model_1 \
    --input_file data/processed/test.txt \
    --output_file results/model_1_predictions.txt
~~~
Replace model1 in the path with model2, model3, etc. depending on which fine-tuned model you want to use, and point --adapter_path to the matching folder in models/.

Using the custom mBERT2mBERT model (best performance)

This model is self-contained inside the mBERT2mBERT/ folder, with its own model.py and inference.py:
~~~
python mBERT2mBERT/inference.py \
    --model_path mBERT2mBERT \
    --input_file data/processed/test.txt \
    --output_file results/mbert2mbert_predictions.txt
~~~

Fine-tuning a model yourself (optional)

If you want to reproduce the fine-tuning instead of using the provided checkpoints:
~~~
python scripts/model1/model.py \
    --model_name_or_path <pretrained-model> \
    --train_file data/processed/train.txt \
    --output_dir models/model_1 \
    --num_train_epochs <n>
~~~

