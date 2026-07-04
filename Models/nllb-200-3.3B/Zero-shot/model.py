import torch
import evaluate
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
# ==========================================
# 1. Configuration & Loading
# ==========================================
token=""
model_id = "facebook/nllb-200-3.3B" 

print("Loading NLLB model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="mya_Mymr")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

# ==========================================
# 2. Data Loading
# ==========================================
def load_parallel_data(src_path, tgt_path):
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        src_lines = [line.strip() for line in f_src.readlines()]
        tgt_lines = [line.strip() for line in f_tgt.readlines()]
    assert len(src_lines) == len(tgt_lines), "Source and target files must match."
    return {"burmese": src_lines, "english": tgt_lines}

print("Loading datasets...")
valid_data = load_parallel_data("source.txt", "target.txt")

# ==========================================
# 3. Inference and Generation
# ==========================================
predictions = []
references = [[tgt] for tgt in valid_data["english"]] 

print(f"Translating {len(valid_data['burmese'])} validation samples...")

# Fetch the exact token ID for English to force the model to output English
forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

for src_text in tqdm(valid_data["burmese"]):
    
    # We do NOT use a few-shot prompt here. Just pass the raw Burmese text.
    inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn"), # Crucial: tells NLLB to translate to English
            max_new_tokens=128, 
            num_beams=4 
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    predictions.append(translation)

# ==========================================
# 4. Calculate BLEU Score
# ==========================================
print("Calculating BLEU score...")
metric = evaluate.load("sacrebleu")
results = metric.compute(predictions=predictions, references=references)

print("\n=== NLLB Validation Results ===")
print(f"SacreBLEU Score: {results}")

bertscore_metric = evaluate.load("bertscore")
bertscore_results = bertscore_metric.compute(
    predictions=predictions, 
    references=references, 
    lang="en"
)
test_bertscore = np.mean(bertscore_results["f1"])
print(f"BERTScore F1: {test_bertscore}")
chrf_metric      = evaluate.load("chrf")
ROUGE_TYPES      = ["rouge2", "rouge3", "rouge4"]

chrf_result = chrf_metric.compute(predictions=predictions, references=references, word_order=2)
chrf_score = chrf_result["score"]
print(f"chrF Score:{chrf_score}")

rouge_metric = evaluate.load("rouge")

# Compute ROUGE for all specified types at once
rouge_results = rouge_metric.compute(
    predictions=predictions,
    references=references,
    rouge_types=ROUGE_TYPES,
    use_aggregator=True   # returns averaged scores across all examples
)

# Print each ROUGE score (F1‑score by default)
for rouge_type in ROUGE_TYPES:
    # The returned dictionary keys are like 'rouge2', 'rouge3', 'rouge4'
    score = rouge_results[rouge_type]
    print(f"{rouge_type.upper()} Score (F1): {score:.4f}")

print("\nSample Outputs:")
for i in range(min(3, len(predictions))):
    print(f"Source:  {valid_data['burmese'][i]}")
    print(f"Target:  {valid_data['english'][i]}")
    print(f"Predict: {predictions[i]}\n")
