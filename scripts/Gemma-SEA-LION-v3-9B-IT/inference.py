import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_ID = "aisingapore/Gemma-SEA-LION-v3-9B-IT"
ADAPTER_PATH  = "."   # folder containing adapter_config.json, adapter_model.safetensors
                      # change to "Models/gemma/Finetuned" if running from repo root
MAX_SEQ_LEN   = 512
MAX_NEW_TOKENS = 192

# HF_TOKEN must be set as an environment variable — never hardcode it.
# export HF_TOKEN="your_token_here"   (in your shell, before running this script)
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# LOAD TOKENIZER, BASE MODEL, LORA ADAPTER
# ─────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
tokenizer.padding_side = "left"   # left-padding for batch generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (9B params — requires a capable GPU, ~18GB+ VRAM in bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=HF_TOKEN,
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


def make_prompt(burmese: str) -> str:
    """Same instruction template used during training."""
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


def translate(text: str) -> str:
    """Translate a single Burmese sentence into English."""
    prompt = make_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    # Model outputs only the translation line, per the prompt instruction
    return decoded.split("\n")[0].strip()


# ─────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if HF_TOKEN is None:
        print("Warning: HF_TOKEN environment variable is not set.")
        print("If the base model is gated/private, set it first:")
        print('  export HF_TOKEN="your_token_here"\n')

    print("Gemma SEA-LION Burmese → English Translator (LoRA finetuned)")
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