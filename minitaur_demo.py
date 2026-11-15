"""Load Minitaur-8B on one GPU and run the Centaur repo's demo prompt (run_minimal.py)."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "marcelbinz/Llama-3.1-Minitaur-8B"
CHOICES = ["H", "Y", "E"]

PROMPT = (
    "You will be presented with triplets of objects, which will be assigned to the keys H, Y, and E.\n"
    "In each trial, please indicate which object you think is the odd one out by pressing the corresponding key.\n"
    "In other words, please choose the object that is the least similar to the other two.\n\n"
    "H: plant, Y: chainsaw, and E: periscope. You press <<H>>.\n"
    "H: tostada, Y: leaf, and E: sail. You press <<H>>.\n"
    "H: clock, Y: crystal, and E: grate. You press <<Y>>.\n"
    "H: barbed wire, Y: kale, and E: sweater. You press <<E>>.\n"
    "H: raccoon, Y: toothbrush, and E: ice. You press <<"
)

print("GPU:", torch.cuda.get_device_name(0))

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
torch.cuda.synchronize()
print(f"Load time: {time.time() - t0:.1f} s")

inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
prompt_ids = inputs["input_ids"][0].tolist()

# How does the tokenizer encode each letter right after "<<"?
# Tokenize prompt + letter and take the token(s) that follow the prompt tokens.
choice_ids = {}
for letter in CHOICES:
    full_ids = tokenizer(PROMPT + letter)["input_ids"]
    same_prefix = full_ids[: len(prompt_ids)] == prompt_ids
    extra = full_ids[len(prompt_ids):]
    print(f"Encoding of {letter!r} after '<<': ids={extra} tokens={tokenizer.convert_ids_to_tokens(extra)} "
          f"(prompt prefix unchanged: {same_prefix})")
    choice_ids[letter] = extra[0]

# (b) Next-token probabilities from the logits
with torch.no_grad():
    logits = model(**inputs).logits[0, -1].float()
probs = torch.softmax(logits, dim=-1)
print("Next-token probabilities:")
for letter, tid in choice_ids.items():
    print(f"  {letter}: {probs[tid].item():.4f}")
print(f"  sum(H,Y,E): {sum(probs[t].item() for t in choice_ids.values()):.4f}")
top = torch.topk(probs, 5)
print("Top-5 tokens:", [(tokenizer.decode([i]), round(p, 4)) for i, p in zip(top.indices.tolist(), top.values.tolist())])

# (a) One sampled next token, temperature 1.0, max_new_tokens=1 (as in the repo)
with torch.no_grad():
    out = model.generate(**inputs, do_sample=True, temperature=1.0, max_new_tokens=1,
                         pad_token_id=tokenizer.eos_token_id)
sampled = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:])
print(f"Sampled next token: {sampled!r}")

print(f"GPU memory allocated now: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU peak memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
