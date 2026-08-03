import time
import torch
from transformers import AutoTokenizer
import main as m

MODEL = m.FP8_DIR
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
engine = m.QwenEngineFP8Opt(
    MODEL,
    per_row_mlp_dir=None,
    per_row_attention_dir=None,
)
messages = [{"role": "user", "content": "请简要介绍人工智能。"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
input_ids = tokenizer.encode(text, add_special_tokens=False)
engine._reset_cache()
logits = engine.forward(input_ids, past_len=0)
next_id = logits[-1].argmax().view(1, 1)
past_len = len(input_ids)
for _ in range(20):
    logits = engine.forward(next_id, past_len=past_len)
    next_id = logits[-1].argmax().view(1, 1)
    past_len += 1

torch.cuda.synchronize()
steps = 300
t0 = time.perf_counter()
for _ in range(steps):
    logits = engine.forward(next_id, past_len=past_len)
    next_id = logits[-1].argmax().view(1, 1)
    past_len += 1
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"decode_tokens_per_second={steps / elapsed:.6f}")
print(f"milliseconds_per_token={elapsed * 1000 / steps:.6f}")
print(f"allocated_gb={torch.cuda.memory_allocated() / 1e9:.6f}")
print(f"peak_allocated_gb={torch.cuda.max_memory_allocated() / 1e9:.6f}")
