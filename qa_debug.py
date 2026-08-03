"""Debug a few questions"""
import torch
from transformers import AutoTokenizer
import main

MODEL = main.FP8_DIR
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
engine = main.QwenEngineFP8Opt(MODEL, per_row_mlp_dir=None, per_row_attention_dir=None)

bad_questions = [
    "从下面的句子中提取人名：李华和王芳一起去公园。只回答人名。",
    "以下数字中哪些是偶数？2, 3, 5, 8, 10。只回答偶数，用逗号隔开。",
    "请输出数字5到1的倒序，用空格隔开。",
    "地球绕太阳转一圈需要多长时间？只回答数字加单位。",
    "请严格输出三个汉字：测试通过。不要添加其他内容。",
]

for q in bad_questions:
    msgs = [{"role": "user", "content": q}]
    enc = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(enc, add_special_tokens=False)
    seq_len = len(input_ids)

    engine._reset_cache()
    logits = engine.forward(input_ids, past_len=0)
    next_id = logits[-1].argmax().item()
    past_len = seq_len
    tokens = [next_id]

    for _ in range(63):
        logits = engine.forward([next_id], past_len=past_len)
        next_id = logits[-1].argmax().item()
        if next_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
            break
        tokens.append(next_id)
        past_len += 1

    print(f"Q: {q}")
    print(f"  Token IDs: {tokens[:10]}...")
    for i, t in enumerate(tokens[:8]):
        txt = tokenizer.decode([t], skip_special_tokens=False)
        print(f"  [{i}] {t}: {txt!r}")
    full = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"  Full decoded: {full!r}")
    print()
