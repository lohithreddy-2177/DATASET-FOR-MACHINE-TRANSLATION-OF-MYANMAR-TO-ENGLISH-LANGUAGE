import torch
from transformers import EncoderDecoderModel, BertTokenizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "."   # folder containing config.json, model.safetensors, vocab.txt, etc.
                    # change to "Models/mbert2mbert-burmese/Finetuned" if running from repo root
MAX_LENGTH = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# LOAD TOKENIZER & MODEL
# ─────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

print("Loading finetuned mBERT2mBERT model...")
model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()


def translate(text: str) -> str:
    """Translate a single Burmese sentence into English."""
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_LENGTH,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True,
            no_repeat_ngram_size=3,
            decoder_start_token_id=model.config.decoder_start_token_id,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
        )

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


# ─────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nmBERT2mBERT Burmese → English Translator (finetuned, frozen encoder/decoder)")
    print("Type a Burmese sentence and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Burmese > ").strip()
        if text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not text:
            continue

        translation = translate(text)
        print(f"English > {translation}\n")