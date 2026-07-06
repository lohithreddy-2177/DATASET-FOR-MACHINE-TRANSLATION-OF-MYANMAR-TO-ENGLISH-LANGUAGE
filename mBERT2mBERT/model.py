from transformers import EncoderDecoderModel, BertTokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from collections import Counter
from torch.optim import AdamW
import json


# ============================
# CONFIGURATION
# ============================
class Config:
    BATCH_SIZE = 32
    EPOCHS = 37
    LEARNING_RATE = 9e-5
    MAX_LENGTH = 100
    WARMUP_STEPS = 2000
    GRADIENT_ACCUMULATION_STEPS = 2
    PATIENCE = 5  # Early stopping patience
    FREEZE_ENCODER = True
    FREEZE_DECODER = True
    TRAIN_CROSS_ATTENTION = True


config = Config()
# ============================
# TOKENIZER AND MODEL SETUP
# ============================
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

print("Loading model...")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-multilingual-cased',
    'bert-base-multilingual-cased'
)

# Configure generation parameters
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.config.encoder.vocab_size


def setup_parameter_freezing(model, config):
    """Properly freeze/unfreeze model parameters"""
    trainable_params = []

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on configuration
    if not config.FREEZE_ENCODER:
        for param in model.encoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("Encoder parameters are trainable")

    if not config.FREEZE_DECODER:
        for param in model.decoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("Decoder parameters are trainable")

    # Unfreeze cross-attention layers
    if config.TRAIN_CROSS_ATTENTION:
        # Look for cross-attention parameters
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in ['cross', 'attention']):
                param.requires_grad = True
                trainable_params.append(param)

        print(f"Training {len(trainable_params)} cross-attention related parameters")

    return trainable_params


trainable_params = setup_parameter_freezing(model, config)


class EfficientTranslationDataset(Dataset):
    def __init__(self, source_path, target_path, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print("Loading text files into memory...")
        with open(source_path, 'r', encoding='utf-8') as f:
            self.source_lines = [line.strip() for line in f]

        with open(target_path, 'r', encoding='utf-8') as f:
            self.target_lines = [line.strip() for line in f]

        assert len(self.source_lines) == len(
            self.target_lines), "Source and target files must have the same number of lines!"
        self.num_samples = len(self.source_lines)
        print(f"Loaded {self.num_samples} translation pairs successfully.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        source_text = self.source_lines[idx]
        target_text = self.target_lines[idx]

        source_encodings = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encodings['input_ids'].squeeze(0),
            'attention_mask': source_encodings['attention_mask'].squeeze(0),
            'labels': target_encodings['input_ids'].squeeze(0)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, config, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Replace pad_token_id with -100 for loss calculation
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'lr': scheduler.get_last_lr()[0]
        })

    return total_loss / len(dataloader)


# ============================
# GEOMETRIC MEAN BLEU CALCULATION
# ============================
def calculate_geometric_bleu(predictions, references):
    """Calculate BLEU score using geometric mean of n-gram precisions"""
    if not predictions or not references:
        return {'bleu': 0}

    # Tokenize
    tokenized_preds = []
    tokenized_refs = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        if pred_tokens or ref_tokens:
            tokenized_preds.append(pred_tokens)
            tokenized_refs.append([ref_tokens])  # Wrap in list for multiple references support

    if not tokenized_preds:
        return {'bleu': 0}

    # Import nltk BLEU scorer
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        chencherry = SmoothingFunction()

        # Calculate BLEU score with geometric mean
        bleu_score = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=(0.25, 0.25, 0.25, 0.25),  # Equal weights for 1-4 grams
            smoothing_function=chencherry.method1
        )

    except ImportError:
        print("NLTK not installed, falling back to simplified BLEU calculation")
        # Fallback to simplified calculation
        bleu_score = calculate_simple_geometric_bleu(tokenized_preds, tokenized_refs)

    return {'bleu': bleu_score}


def calculate_simple_geometric_bleu(predictions, references_list):
    """Simplified geometric BLEU calculation"""
    max_n = 4
    precisions = []

    # Calculate precision for each n-gram
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred_ngrams = 0

        for pred_tokens, ref_list in zip(predictions, references_list):
            if len(pred_tokens) < n:
                continue

            # Generate n-grams for prediction
            pred_ngrams = [tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)]
            if not pred_ngrams:
                continue

            # Generate n-grams for all references
            ref_ngrams_list = []
            for ref_tokens in ref_list:
                ref_ngrams = [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]
                ref_ngrams_list.append(ref_ngrams)

            # Count matches using closest reference length (modified precision)
            for ngram in pred_ngrams:
                # Count in prediction
                pred_count = pred_ngrams.count(ngram)
                # Maximum count in any reference
                max_ref_count = max([ref_ngrams.count(ngram) for ref_ngrams in ref_ngrams_list])

                total_matches += min(pred_count, max_ref_count)
            total_pred_ngrams += len(pred_ngrams)

        if total_pred_ngrams > 0:
            precisions.append(total_matches / total_pred_ngrams)
        else:
            precisions.append(0)

    # Brevity penalty
    c = sum(len(p) for p in predictions)
    r = min(sum(len(r[0]) for r in references_list), c)  # Use closest reference length

    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0

    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bleu_score = bp * geometric_mean

    return bleu_score


# ============================
# VALIDATION FUNCTION
# ============================
def validate(model, dataloader, device, tokenizer, config, split_name="Validation", compute_bleurt=False):
    """Validation with optimized generation.

    compute_bleurt: BLEURT is expensive, so by default it is SKIPPED during
    per-epoch validation. Pass compute_bleurt=True (used for the final test
    evaluation) to actually run it.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=split_name):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Calculate loss
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_loss,
                return_dict=True
            )
            total_loss += outputs.loss.item()

            # Generate predictions with optimized settings
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.MAX_LENGTH,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                no_repeat_ngram_size=3,
                decoder_start_token_id=model.config.decoder_start_token_id,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id
            )

            # Decode batch efficiently
            pred_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            ref_texts = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            all_predictions.extend([t.strip() for t in pred_texts])
            all_references.extend([t.strip() for t in ref_texts])

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=all_predictions,
        references=all_references,
        lang="en"
    )
    bertscoreF1 = np.mean(bertscore_results["f1"])
    bleu_scores = calculate_geometric_bleu(all_predictions, all_references)
    chrf_metric = evaluate.load("chrf")
    ROUGE_TYPES = ["rouge2", "rouge3", "rouge4"]
    chrf_result = chrf_metric.compute(predictions=all_predictions, references=all_references, word_order=2)
    chrf_score = chrf_result["score"]
    print(f"chrF Score:{chrf_score}")

    rouge_metric = evaluate.load("rouge")

    # Compute ROUGE for all specified types at once
    rouge_results = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references,
        rouge_types=ROUGE_TYPES,
        use_aggregator=True  # returns averaged scores across all examples
    )

    # Print each ROUGE score (F1‑score by default)
    for rouge_type in ROUGE_TYPES:
        # The returned dictionary keys are like 'rouge2', 'rouge3', 'rouge4'
        score = rouge_results[rouge_type]
        print(f"{rouge_type.upper()} Score (F1): {score:.4f}")

    # BLEURT is slow, so only compute it when explicitly requested
    # (i.e. for the final test-set evaluation, not every epoch's validation)
    if compute_bleurt:
        bleurt_metric = evaluate.load("bleurt", config_name="BLEURT-20")
        bleurt_results = bleurt_metric.compute(
            predictions=all_predictions,
            references=all_references,
        )
        bleurt_score = float(np.mean(bleurt_results["scores"]))
    else:
        bleurt_score = None

    avg_loss = total_loss / len(dataloader)
    return avg_loss, bleu_scores, bertscoreF1, chrf_score, rouge_results, bleurt_score, all_predictions, all_references


print("\nCreating dataset...")
dataset = EfficientTranslationDataset(
    source_path="data/source.txt",
    target_path="data/target.txt",
    tokenizer=tokenizer,
    max_length=config.MAX_LENGTH
)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\nUsing device: {device}")

# Optimizer and scheduler
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(
    trainable_parameters,
    lr=config.LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Learning rate scheduler
total_steps = len(train_dataloader) * config.EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
warmup_steps = min(config.WARMUP_STEPS, total_steps // 10)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print("\n" + "=" * 50)
print("STARTING TRAINING")
print("=" * 50)

best_bleu = 0
patience_counter = 0
training_history = []

# Store losses for plotting
train_losses = []
val_losses = []
val_bleus = []
val_bert = []
for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
    print("-" * 50)

    # Training
    train_loss = train_epoch(
        model, train_dataloader, optimizer, scheduler,
        device, epoch, config, config.GRADIENT_ACCUMULATION_STEPS
    )
    train_losses.append(train_loss)

    # BLEURT is skipped here (compute_bleurt=False) since it's expensive;
    # it is only computed once, at the very end, on the test set.
    val_loss, bleu_scores, bertscore, chrf_score, rouge_results, bleurt_score, _, _ = validate(
        model, val_dataloader, device, tokenizer, config, "Validation", compute_bleurt=False
    )
    val_losses.append(val_loss)
    val_bleus.append(bleu_scores['bleu'])
    val_bert.append(bertscore)

    epoch_stats = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_bleu': bleu_scores['bleu'],
        'val_bert': bertscore,
        'val_chrf': chrf_score,
        'val_rouge': rouge_results,
        'val_bleurt': bleurt_score  # will be None for every epoch (not computed)
    }
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(epoch_stats) + "\n")
    training_history.append(epoch_stats)

    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BLEU: {bleu_scores['bleu']:.4f}")
    print(epoch_stats)

    if bleu_scores['bleu'] > best_bleu:
        best_bleu = bleu_scores['bleu']
        os.makedirs("best_model", exist_ok=True)
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")

        # Also save training history
        torch.save(training_history, "best_model/training_history.pt")

        print(f"  ✓ New best model! BLEU: {bleu_scores['bleu']:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{config.PATIENCE}")

    if patience_counter >= config.PATIENCE:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        break

print("\n" + "=" * 50)
print("FINAL TEST EVALUATION")
print("=" * 50)

print("Loading best model for testing...")
model = EncoderDecoderModel.from_pretrained("best_model")
model = model.to(device)

# Only here, at the very end, on the test set, do we compute BLEURT
# (compute_bleurt=True)
test_loss, test_bleu_scores, test_bert, test_chrf, test_rouge, test_bleurt, test_predictions, test_references = validate(
    model, test_dataloader, device, tokenizer, config, "Test", compute_bleurt=True
)

print(f"\nTest Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test BLEU: {test_bleu_scores['bleu']:.4f}")
print(f"Test BERTScore: {test_bert}")
print(f"Test chrf score: {test_chrf}")
print(f"Test Rouge : {test_rouge}")
print(f"Test BLEURT SCORE : {test_bleurt}")

test_results = {
    'test_loss': test_loss,
    'test_bleu': test_bleu_scores['bleu'],
    'test_bert': test_bert,
    "Test chrf score": test_chrf,
    "Test Rouge ": test_rouge,
    'test_bluert score': test_bleurt,
    'predictions': test_predictions[:10],
    'references': test_references[:10]
}
print(test_results)
with open("results.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(test_results) + "\n")
torch.save(test_results, "best_model/test_results.pt")

print("\nSample predictions:")
for i in range(min(5, len(test_predictions))):
    print(f"  Reference: {test_references[i]}")
    print(f"  Prediction: {test_predictions[i]}")
    print()

# ============================
# PLOTTING METRICS
# ============================
try:
    epochs_range = range(1, len(train_losses) + 1)

    # Plot 1: Training & Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

    # Plot 2: Validation BLEU Score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, val_bleus, 'g-', label='Validation BLEU')
    plt.title('Validation BLEU Score Trend')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_summary.png')
    plt.close()
    print("\n[Success] Plots saved as 'training_history.png' and 'final_summary.png'")
except Exception as e:
    print(f"\n[Warning] Could not generate plots: {e}")

# ============================
# FINAL SUMMARY
# ============================
print("\n" + "=" * 50)
print("TRAINING COMPLETED - FINAL SUMMARY")
print("=" * 50)
print(f"Total epochs trained: {len(training_history)}")
print(f"\nBest Validation BLEU: {best_bleu:.4f}")
print(f"Final Test BLEU: {test_results['test_bleu']:.4f}")
print(f"Final Test Loss: {test_results['test_loss']:.4f}")

# Save final model
os.makedirs("final_model", exist_ok=True)
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")

# Save complete training history
final_history = {
    'config': config.__dict__,
    'training_history': training_history,
    'test_results': test_results,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_bleus': val_bleus
}

torch.save(final_history, "final_model/complete_history.pt")
print("\nFinal model and complete history saved to 'final_model/'")
print("Plots saved as 'training_history.png' and 'final_summary.png'")
