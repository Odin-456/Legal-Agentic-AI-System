import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


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
print("LLAMA PROPERTY-LAW TEST")
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
# Test question
# ============================================================

question = """
Under the Residential Tenancies Act 2010 (NSW), what remedies
may be available to a tenant when a landlord fails to carry out
required repairs?
"""


# ============================================================
# Create Llama chat prompt
# ============================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant answering questions about "
            "Australian law. Answer clearly and cautiously. "
            "Do not invent legislation sections or legal "
            "authorities. If you are uncertain, say so."
        ),
    },
    {
        "role": "user",
        "content": question,
    },
]


print("Creating prompt...")

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# ============================================================
# Tokenize prompt
# ============================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs["attention_mask"].to(DEVICE)

print("Generating answer...")
print()


# ============================================================
# Generate
# ============================================================

with torch.no_grad():

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


# ============================================================
# Extract only generated tokens
# ============================================================

generated_tokens = outputs[
    0,
    input_ids.shape[-1]:
]


# ============================================================
# Decode answer
# ============================================================

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)


# ============================================================
# Display result
# ============================================================

print("=" * 70)
print("QUESTION")
print("=" * 70)

print(question.strip())

print()

print("=" * 70)
print("LLAMA ANSWER")
print("=" * 70)

print(answer.strip())

print()

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
