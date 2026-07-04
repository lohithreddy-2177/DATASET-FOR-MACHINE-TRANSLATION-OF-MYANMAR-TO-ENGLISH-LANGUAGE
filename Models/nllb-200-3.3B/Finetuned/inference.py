import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "lohith1150/nllb-3.3B-myan-fine-tuned"
BASE_MODEL = "facebook/nllb-200-3.3B"

print("Loading tokenizer from HF Hub...")
tokenizer = AutoTokenizer.from_pretrained(
    REPO_ID, src_lang="mya_Mymr", tgt_lang="eng_Latn"
)

print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

print("Loading LoRA adapter from HF Hub...")
model = PeftModel.from_pretrained(base_model, REPO_ID)
model.to(DEVICE)
model.eval()

test_sentence = input("Enter your myan sentence:")

inputs = tokenizer(test_sentence, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_new_tokens=198,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\nSource     : {test_sentence}")
print(f"Translation: {translation}")
