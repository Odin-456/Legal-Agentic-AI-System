import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "isaacus/open-australian-legal-llm"


# ============================================================
# Select device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


print("=" * 70)
print("OPEN AUSTRALIAN LEGAL LLM TEST")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print()


# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")
print()


# ============================================================
# Load model
# ============================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
)

model = model.to(DEVICE)
model.eval()

print("Model loaded.")
print()


# ============================================================
# Question
# ============================================================

question = (
    "Under the Residential Tenancies Act 2010 (NSW), "
    "what remedies may be available to a tenant when a "
    "landlord fails to carry out required repairs?"
)


# ============================================================
# Simple prompt
# ============================================================

prompt = f"""Australian legal question:

{question}

Answer:
"""


# ============================================================
# Tokenize
# ============================================================

print("Generating answer...")
print()

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
)

input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs["attention_mask"].to(DEVICE)


# ============================================================
# Generate
# ============================================================

with torch.no_grad():

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,

        max_new_tokens=250,

        do_sample=False,

        repetition_penalty=1.2,

        no_repeat_ngram_size=3,

        pad_token_id=tokenizer.eos_token_id,
    )


# ============================================================
# Extract generated tokens
# ============================================================

generated_tokens = outputs[
    0,
    input_ids.shape[-1]:
]


# ============================================================
# Decode
# ============================================================

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)


# ============================================================
# Display
# ============================================================

print("=" * 70)
print("QUESTION")
print("=" * 70)

print(question)

print()

print("=" * 70)
print("OPEN AUSTRALIAN LEGAL LLM ANSWER")
print("=" * 70)

print(answer.strip())

print()

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

