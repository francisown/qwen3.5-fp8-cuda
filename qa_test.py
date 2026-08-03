"""详细中文问答测试"""
import torch, time
from transformers import AutoTokenizer
import main

MODEL = main.FP8_DIR
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
engine = main.QwenEngineFP8Opt(MODEL, per_row_mlp_dir=None, per_row_attention_dir=None)

questions = [
    # 常识
    "中国的首都是哪里？只回答城市名。",
    "世界上最高的山峰是什么？只回答山峰名称。",
    "一年有多少个月？只回答数字。",
    # 数学
    "小明有17个苹果，送给朋友5个，又买了8个，现在有多少个？只回答数字。",
    "一个长方形的长是8厘米，宽是5厘米，它的面积是多少平方厘米？只回答数字。",
    "3乘以7加上4等于多少？只回答数字。",
    "100减37等于多少？只回答数字。",
    # 逻辑
    "所有猫都是动物，小花是一只猫。小花是不是动物？只回答\u201c是\u201d或\u201c否\u201d。",
    "如果今天是星期一，那么3天后是星期几？只回答星期几。",
    # 提取 & 指令
    "从下面的句子中提取人名：李华和王芳一起去公园。只回答人名。",
    "以下数字中哪些是偶数？2, 3, 5, 8, 10。只回答偶数，用逗号隔开。",
    "请输出数字5到1的倒序，用空格隔开。",
    "把这句话翻译成英文：你好，世界。只回答英文。",
    # 开放知识
    "鲁迅的《狂人日记》是中国现代文学的第一篇白话小说吗？只回答\u201c是\u201d或\u201c否\u201d。",
    "地球绕太阳转一圈需要多长时间？只回答数字加单位。",
]

for q in questions:
    msgs = [{"role": "user", "content": q}]
    enc = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(enc, add_special_tokens=False)
    seq_len = len(input_ids)

    engine._reset_cache()
    logits = engine.forward(input_ids, past_len=0)
    next_id = logits[-1].argmax().item()
    past_len = seq_len
    tokens = [next_id]

    t0 = time.perf_counter()
    for _ in range(63):
        logits = engine.forward([next_id], past_len=past_len)
        next_id = logits[-1].argmax().item()
        if next_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
            break
        tokens.append(next_id)
        past_len += 1
    elapsed = time.perf_counter() - t0

    # 只取第一个换行之前的内容（干净的答案）
    full = tokenizer.decode(tokens, skip_special_tokens=True)
    # clean: take text until first newline or <think>
    clean = full.split("\n")[0].strip()
    # also stop at <think>
    if "<think>" in clean:
        clean = clean.split("<think>")[0].strip()

    tok_s = len(tokens) / elapsed
    print(f"Q: {q}")
    print(f"A: {clean}  ({tok_s:.0f} tok/s)")
    print()
