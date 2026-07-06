import os
import json
import math
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

# ==========================================
# 1. Configuration
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
MODEL_ID      = "aisingapore/Gemma-SEA-LION-v3-9B-IT"
OUTPUT_DIR    = "./lora_checkpoints"
RESULTS_PATH  = "evaluation_results_lora.json"

# LoRA hyper-parameters
LORA_R        = 16       # rank — increase to 32/64 for more capacity
LORA_ALPHA    = 32       # scaling factor (typically 2× rank)
LORA_DROPOUT  = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyper-parameters
TRAIN_SPLIT   = 0.9      # 90 % train / 10 % validation
BATCH_SIZE    = 2        # per-device; increase if VRAM allows
GRAD_ACCUM    = 8        # effective batch = BATCH_SIZE × GRAD_ACCUM × n_gpus
EPOCHS        = 5
LR            = 2e-4
MAX_SEQ_LEN   = 512      # prompt + completion; truncated if longer
WARMUP_RATIO  = 0.05

# Inference
INF_BATCH_SIZE   = 4
MAX_NEW_TOKENS   = 192

# ==========================================
# 2. Load tokenizer & model
# ==========================================
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.padding_side  = "right"   # right-pad during training
tokenizer.truncation_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)
model.config.use_cache = False   # required for gradient-checkpointing

# ==========================================
# 3. Attach LoRA adapters
# ==========================================
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
)
model = get_peft_model(model, lora_cfg)

# Fix: gradient checkpointing with PEFT requires input embeddings to have
# requires_grad=True, otherwise the backward pass has no grad_fn on the
# first tensor and raises "does not require grad".
model.enable_input_require_grads()

model.print_trainable_parameters()

# ==========================================
# 4. Data loading & tokenisation
# ==========================================
def load_parallel(src_path: str, tgt_path: str):
    with open(src_path, encoding="utf-8") as fs, \
         open(tgt_path, encoding="utf-8") as ft:
        src = [l.strip() for l in fs]
        tgt = [l.strip() for l in ft]
    assert len(src) == len(tgt), "Source / target line counts differ."
    return src, tgt

def make_prompt(burmese: str) -> str:
    """Instruction prompt (no few-shot examples — the LoRA learns from data)."""
    messages = [
        {
            "role": "user",
            "content": (
                "You are a professional translator. "
                "Translate the following Burmese text to English. "
                "Output only the English translation, nothing else.\n\n"
                f"Burmese: {burmese}\nEnglish:"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def tokenise_row(row):
    """
    Create input_ids with the completion appended.
    Only the completion tokens contribute to the loss (labels for the prompt
    are masked to -100).
    """
    prompt      = make_prompt(row["burmese"])
    completion  = row["english"] + tokenizer.eos_token

    prompt_ids      = tokenizer(prompt,     add_special_tokens=False)["input_ids"]
    completion_ids  = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels    = [-100] * len(prompt_ids) + completion_ids

    # Truncate from the LEFT (keep completion intact)
    if len(input_ids) > MAX_SEQ_LEN:
        excess      = len(input_ids) - MAX_SEQ_LEN
        input_ids   = input_ids[excess:]
        labels      = labels[excess:]

    return {"input_ids": input_ids, "labels": labels}


print("Loading data …")
burmese_lines, english_lines = load_parallel("source.txt", "target.txt")

raw_dataset = Dataset.from_dict({"burmese": burmese_lines, "english": english_lines})

# Train / validation split (stratified by length is overkill here)
split       = raw_dataset.train_test_split(test_size=1 - TRAIN_SPLIT, seed=42)
train_ds    = split["train"].map(tokenise_row, remove_columns=["burmese", "english"])
val_ds      = split["test"]          # keep raw strings for generation-based eval

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ==========================================
# 5. Training
# ==========================================
collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch",        # safe default; swap to paged_adamw_8bit if VRAM is tight
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collator,
)

print("\n=== Starting LoRA fine-tuning ===")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Adapters saved to {OUTPUT_DIR}")

# ==========================================
# 6. Reload model for inference
#    (merge LoRA weights for faster generation)
# ==========================================
print("\nMerging LoRA adapters for inference …")
del model          # free GPU memory
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()
merged_model.eval()

# Restore right-padding → left-padding for batch inference
tokenizer.padding_side = "left"

# ==========================================
# 7. Inference on validation set
# ==========================================
val_burmese  = val_ds["burmese"]
val_english  = val_ds["english"]

predictions  = []
references   = list(val_english)

print(f"\nTranslating {len(val_burmese)} validation samples …")
for i in tqdm(range(0, len(val_burmese), INF_BATCH_SIZE)):
    batch_src = val_burmese[i : i + INF_BATCH_SIZE]
    prompts   = [make_prompt(src) for src in batch_src]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(merged_model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[:, input_len:]
    decoded   = tokenizer.batch_decode(generated, skip_special_tokens=True)

    for text in decoded:
        predictions.append(text.split("\n")[0].strip().rstrip("\r"))

    if torch.cuda.is_available() and (i // INF_BATCH_SIZE) % 10 == 0:
        torch.cuda.empty_cache()

# ==========================================
# 8. Metrics (identical to few-shot script)
# ==========================================
print("\nCalculating metrics …")

# BLEU
bleu_metric  = evaluate.load("sacrebleu")
bleu_result  = bleu_metric.compute(
    predictions=predictions,
    references=[[r] for r in references],
)

# BERTScore
bertscore_metric  = evaluate.load("bertscore")
bertscore_results = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="en",
)

# chrF++
chrf_metric  = evaluate.load("chrf")
chrf_result  = chrf_metric.compute(
    predictions=predictions,
    references=[[r] for r in references],
    word_order=2,
)

# ROUGE-1 / ROUGE-2
rouge_metric   = evaluate.load("rouge")
rouge_results  = rouge_metric.compute(
    predictions=predictions,
    references=references,
    rouge_types=["rouge1", "rouge2"],
    use_aggregator=True,
)

# BLEURT-20
print("Loading BLEURT-20 (downloads checkpoint on first run, ~1.5 GB) …")
bleurt_metric  = evaluate.load("bleurt", config_name="BLEURT-20")
bleurt_results = bleurt_metric.compute(
    predictions=predictions,
    references=references,
)
bleurt_score = float(np.mean(bleurt_results["scores"]))

# ==========================================
# 9. Print Results
# ==========================================
print(f"\n=== {MODEL_ID} LoRA Fine-tuned Results ===")
print(f"BLEU       : {bleu_result['score']:.4f}")
print(f"BERTScore  : {np.mean(bertscore_results['f1']):.4f}")
print(f"chrF++     : {chrf_result['score']:.4f}")
for rt in ["rouge1", "rouge2"]:
    print(f"{rt.upper()}      : {rouge_results[rt]:.4f}")
print(f"BLEURT-20  : {bleurt_score:.4f}")

# ==========================================
# 10. Persist Results
# ==========================================
output = {
    "model": MODEL_ID,
    "adapter": OUTPUT_DIR,
    "num_train_samples": len(train_ds),
    "num_val_samples": len(val_ds),
    "lora_config": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "epochs": EPOCHS,
        "lr": LR,
    },
    "metrics": {
        "bleu":           round(bleu_result["score"], 4),
        "bertscore_f1":   round(float(np.mean(bertscore_results["f1"])), 4),
        "chrf_plus_plus": round(chrf_result["score"], 4),
        "rouge1":         round(rouge_results["rouge1"], 4),
        "rouge2":         round(rouge_results["rouge2"], 4),
        "bleurt_20":      round(bleurt_score, 4),
    },
    "samples": [
        {
            "source":     val_burmese[i],
            "reference":  references[i],
            "prediction": predictions[i],
        }
        for i in range(min(5, len(predictions)))
    ],
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")

print("\nSample Outputs:")
for item in output["samples"]:
    print(f"  Source : {item['source']}")
    print(f"  Target : {item['reference']}")
    print(f"  Predict: {item['prediction']}")
    print()