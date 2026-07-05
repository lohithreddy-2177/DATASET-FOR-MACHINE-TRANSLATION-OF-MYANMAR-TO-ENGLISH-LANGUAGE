import os
import torch
import numpy as np
import sacrebleu
import evaluate
import random
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    EarlyStoppingCallback,
)

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
TRAIN_RATIO  = 0.8
VALID_RATIO  = 0.1
TEST_RATIO   = 0.1
MAX_SRC_LEN  = 128
MAX_TGT_LEN  = 198
MODEL_NAME   = "facebook/nllb-200-3.3B"
OUTPUT_DIR   = "./nllb_lora_optimized"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

# Ensure reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# 1. TOKENIZER & MODEL
# ─────────────────────────────────────────────
print("\n[1/5] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, src_lang="mya_Mymr", tgt_lang="eng_Latn"
)

# Load in fp16 to save memory
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Always decode to English (use the official language code ID for safety)
model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

# ─────────────────────────────────────────────
# 2. PEFT  —  LoRA (Expanded for better alignment)
# ─────────────────────────────────────────────
print("\n[2/5] Applying LoRA across model structure...")

lora_config = LoraConfig(
    r                  = 32,
    lora_alpha         = 64,
    target_modules     = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout       = 0.1,
    bias               = "none",
    task_type          = "SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#model.gradient_checkpointing_enable()

# ─────────────────────────────────────────────
# ALIGN GENERATION CONFIGURATION
# ─────────────────────────────────────────────
# Override the generation config so that both training/validation
# and final testing use exactly the same decoding strategy.
model.generation_config.num_beams            = 5
model.generation_config.repetition_penalty   = 1.0
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.early_stopping       = True
model.generation_config.length_penalty       = 1.0
model.generation_config.max_new_tokens = MAX_TGT_LEN   # guarantees 198 NEW tokens
# remove max_length from generation_config

# ─────────────────────────────────────────────
# 3. DATA  —  load, shuffle, split, tokenize
# ─────────────────────────────────────────────
print("\n[3/5] Loading and preparing data...")

def load_raw(src_path, tgt_path):
    """Read parallel text files and return cleaned (src, tgt) line lists."""
    with open(src_path, "r", encoding="utf-8") as fs, \
         open(tgt_path, "r", encoding="utf-8") as ft:
        pairs = [
            (s.strip(), t.strip())
            for s, t in zip(fs, ft)
            if s.strip() and t.strip()
        ]
    src_lines = [p[0] for p in pairs]
    tgt_lines = [p[1] for p in pairs]
    raw_count = sum(1 for _ in open(src_path, encoding="utf-8"))
    print(f"  Raw lines         : {raw_count:>10,}")
    print(f"  After cleaning    : {len(src_lines):>10,}")
    return src_lines, tgt_lines

def tokenize_pairs(src_lines, tgt_lines):
    # Tokenize source
    model_inputs = tokenizer(
        list(src_lines),
        max_length=MAX_SRC_LEN,
        truncation=True,
        padding=False,
    )
    # Tokenize target with its own max_length using text_target trick
    labels = tokenizer(
        text_target=list(tgt_lines),
        max_length=MAX_TGT_LEN,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(model_inputs)

src_lines, tgt_lines = load_raw("source.txt", "target.txt")
total = len(src_lines)

# --- Shuffle the data before splitting ---
pairs = list(zip(src_lines, tgt_lines))
random.shuffle(pairs)
src_lines, tgt_lines = zip(*pairs)

train_end = int(total * TRAIN_RATIO)
valid_end = int(total * (TRAIN_RATIO + VALID_RATIO))

train_src, train_tgt = src_lines[:train_end],        tgt_lines[:train_end]
valid_src, valid_tgt = src_lines[train_end:valid_end], tgt_lines[train_end:valid_end]
test_src,  test_tgt  = src_lines[valid_end:],          tgt_lines[valid_end:]

print(f"\n  Dataset Split:")
print(f"  ├── Train      : {len(train_src):>7,}")
print(f"  ├── Validation : {len(valid_src):>7,}")
print(f"  └── Test       : {len(test_src):>7,}")

print("\n  Tokenizing splits...")
train_dataset = tokenize_pairs(train_src, train_tgt)
valid_dataset = tokenize_pairs(valid_src, valid_tgt)

# ─────────────────────────────────────────────
# 4. METRICS & CALLBACKS
# ─────────────────────────────────────────────

class SaveEpochResultsCallback(TrainerCallback):
    """Appends epoch, train-loss, eval-loss, and BLEU to a TSV file after every epoch."""
    def __init__(self, filename="results.txt"):
        self.filename = filename
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\teval_loss\tbleu\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = round(state.epoch)
        train_loss = "N/A"
        eval_loss = "N/A"
        bleu = "N/A"
        
        for log in state.log_history:
            if "epoch" in log and round(log["epoch"]) == current_epoch:
                if "loss" in log:
                    train_loss = f"{log['loss']:.4f}"
                if "eval_loss" in log:
                    eval_loss = f"{log['eval_loss']:.4f}"
                if "eval_bleu" in log:
                    bleu = f"{log['eval_bleu']:.4f}"
        
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"{current_epoch}\t{train_loss}\t{eval_loss}\t{bleu}\n")


def compute_metrics(eval_pred):
    """BLEU score computed on the validation split during training."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": round(bleu.score, 4)}


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
print("\n[4/5] Starting training...")

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir                  = "./nllb_lora_optimized",
    logging_strategy            = "epoch",
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    learning_rate               = 2e-5, 
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    weight_decay                = 0.01,  
    num_train_epochs            = 20,
    predict_with_generate       = True,
    fp16                        = torch.cuda.is_available(),
    # generation_max_length and generation_num_beams are no longer needed
    # because we've updated model.generation_config. We keep them here as
    # they are valid arguments, but they will not override the config.
    generation_max_length       = MAX_TGT_LEN,
    generation_num_beams        = 5,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "bleu",
    greater_is_better           = True,
    report_to                   = "none",
)

trainer = Seq2SeqTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_dataset,
    eval_dataset     = valid_dataset,
    processing_class = tokenizer,
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
    callbacks        = [
        SaveEpochResultsCallback("test_results_6.txt"),
        EarlyStoppingCallback(early_stopping_patience=5)   # stop if no BLEU improvement for 3 evals
    ],
)

trainer.train()
trainer.save_model("./nllb_lora_optimized")

# ─────────────────────────────────────────────
# 6. TEST EVALUATION  —  BLEU + Loss + BERTScore
# ─────────────────────────────────────────────
print("\n[5/5] Evaluating on test set...")

model.eval()
model.to(DEVICE)

BATCH_SIZE = 16

def batch_generate(src_batch):
    """Tokenize a list of source sentences and generate translations.
       Now uses the model's generation_config which is identical to training."""
    inputs = tokenizer(
        src_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SRC_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids            = inputs["input_ids"],
            attention_mask       = inputs["attention_mask"],
            forced_bos_token_id  = tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens      = MAX_TGT_LEN 
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# 6a. Generate predictions in batches
predictions = []
for i in tqdm(range(0, len(test_src), BATCH_SIZE), desc="  Generating"):
    batch = test_src[i : i + BATCH_SIZE]
    predictions.extend(batch_generate(batch))

# 6b. BLEU Metric
bleu_result = sacrebleu.corpus_bleu(predictions, [test_tgt])

# 6c. BERTScore Metric
print("  Computing BERTScore...")
bertscore_metric = evaluate.load("bertscore")
bertscore_results = bertscore_metric.compute(
    predictions=predictions,
    references=test_tgt,
    lang="en"
)
test_bertscore = np.mean(bertscore_results["f1"])

chrf_metric      = evaluate.load("chrf")
ROUGE_TYPES      = ["rouge2", "rouge3", "rouge4"]

chrf_result = chrf_metric.compute(predictions=predictions, references=test_tgt, word_order=2)
chrf_score = chrf_result["score"]


rouge_metric = evaluate.load("rouge")

# Compute ROUGE for all specified types at once
rouge_results = rouge_metric.compute(
    predictions=predictions,
    references=test_tgt,
    rouge_types=ROUGE_TYPES,
    use_aggregator=True   # returns averaged scores across all examples
)




# 6d. Cross-entropy loss on test set via trainer wrapper
test_dataset = tokenize_pairs(test_src, test_tgt)
test_results = trainer.predict(test_dataset)
test_loss = test_results.metrics.get("test_loss", None)

print(f"\n  ── Test Results ──────────────────────────")
print(f"  Loss            : {test_loss }" if test_loss is not None else "  Loss            : N/A")
print(f"  BLEU Score      : {bleu_result }")
print(f"  BERTScore (F1)  : {test_bertscore }")
print(f"  chrF Score     :  {chrf_score}" )
for rouge_type in ROUGE_TYPES:
    # The returned dictionary keys are like 'rouge2', 'rouge3', 'rouge4'
    score = rouge_results[rouge_type]
    print(f"{rouge_type.upper()} Score (F1): {score:.4f}")
print(f"  ──────────────────────────────────────────")

# 6e. Save comprehensive test results to file
with open("test_results_6.txt", "w", encoding="utf-8") as f:
    f.write(f"test_loss\t{test_loss}\n")
    f.write(f"test_bleu\t{bleu_result }\n")
    f.write(f"test_bertscore\t{test_bertscore}\n")
    f.write(f"\n{'SOURCE':<60}  {'PREDICTION':<60}  REFERENCE\n")
    f.write("-" * 185 + "\n")
    for src, pred, ref in zip(test_src[10], predictions[10], test_tgt[10]):
        f.write(f"{src:<60}  {pred:<60}  {ref}\n")

print("\n  Per-sample results saved to test_results_6.txt")