from transformers import pipeline
from datasets import load_from_disk

ds = load_from_disk("./datasets")  # 加载测试集
question = ds["test"][5]["messages"][:2]  # 获取第一条测试数据的CoT内容

model = "results/checkpoint-321"
generator = pipeline("text-generation", model=model, device="cuda:2")

output = generator(
    question,
    max_new_tokens=2048,
    return_full_text=False,
)[0]
print(output["generated_text"])
