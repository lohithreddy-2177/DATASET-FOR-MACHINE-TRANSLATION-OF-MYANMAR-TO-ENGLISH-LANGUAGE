import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_NAME = "facebook/nllb-200-3.3B"
ADAPTER_PATH    = "."   # folder containing adapter_config.json, adapter_model.safetensors, tokenizer files
                        # change to "Models/nllb-200-3.3B/Finetuned" if running from repo root
MAX_SRC_LEN     = 128
MAX_TGT_LEN     = 198
SRC_LANG        = "mya_Mymr"   # Myanmar
TGT_LANG        = "eng_Latn"   # English
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# LOAD TOKENIZER, BASE MODEL, LORA ADAPTER
# ─────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH, src_lang=SRC_LANG, tgt_lang=TGT_LANG
)

print("Loading base model (this may take a while the first time)...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(DEVICE)
model.eval()

# Same generation config used during training/testing
model.generation_config.forced_bos_token_id  = tokenizer.convert_tokens_to_ids(TGT_LANG)
model.generation_config.num_beams            = 5
model.generation_config.repetition_penalty   = 1.0
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.early_stopping       = True
model.generation_config.length_penalty       = 1.0
model.generation_config.max_new_tokens       = MAX_TGT_LEN


def translate(text: str) -> str:
    """Translate a single Myanmar sentence into English."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SRC_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids           = inputs["input_ids"],
            attention_mask      = inputs["attention_mask"],
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_new_tokens       = MAX_TGT_LEN,
        )

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ─────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nNLLB Myanmar → English Translator (finetuned with LoRA)")
    print("Type a Myanmar sentence and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Myanmar > ").strip()
        if text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not text:
            continue

        translation = translate(text)
        print(f"English > {translation}\n")
