import itertools
import operator
import json
import random
import os
from tqdm import tqdm
import datasets

# ===================================================================
# 1. 认知叙事器 (Cognitive Narrator)
# ===================================================================
class LongCoTLogger:
    def __init__(self, initial_numbers, pruning_level=1):
        self.thinking_process = []
        self.initial_numbers = tuple(sorted(initial_numbers))
        self.pruning_level = pruning_level
        self.log_start(initial_numbers)

    def _get_prefix(self):
        return "- "

    def log_start(self, numbers):
        self.thinking_process.append(f"开始解决24点问题，初始数字为: {numbers}。目标是得到24。")

    def log_branch_summary_and_backtrack(self, numbers, a, b):
        if self.pruning_level != 1: return
        self.thinking_process.append(f"{self._get_prefix()}[分支探索总结] 对数字 {a} 和 {b} 在当前列表 {numbers} 中的所有运算尝试均无法通向最终解，放弃此分支并回溯。")

    def log_final_solution(self, path):
        self.thinking_process.append(f"\n[成功] 找到一个解！最终的计算步骤是: {path} = 24")

    def log_no_solution(self):
        self.thinking_process.append(f"\n[失败] 穷尽了所有可能性，这组数字 {list(self.initial_numbers)} 没有解。")

    def get_long_cot(self):
        return "\n".join(self.thinking_process)

# ===================================================================
# 2. 回溯求解器 与 表达式生成器
# ===================================================================
def solve_24_points_recursive(numbers, logger):
    if len(numbers) == 1: return abs(numbers[0] - 24) < 1e-6
    for a, b in itertools.permutations(numbers, 2):
        rem = list(numbers); rem.remove(a); rem.remove(b); found = False
        for op, sym in [(operator.add, '+'), (operator.mul, '*'), (operator.sub, '-'), (operator.truediv, '/')]:
            if sym in '+*' and a > b: continue
            if sym == '/' and abs(b) < 1e-6: continue
            if solve_24_points_recursive([op(a, b)] + rem, logger):
                found = True
                if logger.pruning_level == 1: return True
        if logger.pruning_level == 1 and not found: logger.log_branch_summary_and_backtrack(sorted(numbers), a, b)
        if found: return True
    return False

def get_solution_expression_final(numbers):
    if len(numbers) == 1: return f"{numbers[0]:.0f}" if abs(numbers[0] - 24) < 1e-6 else None
    for a, b in itertools.permutations(numbers, 2):
        rem = list(numbers); rem.remove(a); rem.remove(b)
        for op, sym, comm in [(operator.add, '+', True), (operator.mul, '*', True), (operator.sub, '-', False), (operator.truediv, '/', False)]:
            if comm and a > b: continue
            if sym == '/' and abs(b) < 1e-6: continue
            path = get_solution_expression_final([op(a, b)] + rem)
            if path:
                a_str, b_str = (f"{x:.0f}" if x == int(x) else str(x) for x in (a,b))
                return path.replace(f"{op(a,b):.0f}", f"({a_str} {sym} {b_str})", 1)
    return None

def generate_24_point_instance(numbers, pruning_level=1):
    logger = LongCoTLogger(numbers, pruning_level=pruning_level)
    if solve_24_points_recursive(list(numbers), logger):
        expr = get_solution_expression_final(list(numbers))
        logger.log_final_solution(expr if expr else "路径已找到，但表达式生成失败。")
    else: logger.log_no_solution()
    sys_prompt = "你是一个数学解题专家。请解决经典的24点游戏。你需要展示详细的思考过程，包括所有成功的尝试和失败的回溯，最后给出答案。"
    user_prompt = f"请用 {list(numbers)} 这几个数字解决24点问题。"
    # 注意：这里返回的是一个包含 "messages" 键的字典，而不是直接的messages列表
    # 这是因为.from_list()方法期望一个字典列表，其中每个字典代表一行数据。
    return {"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": logger.get_long_cot()}]}

# ===================================================================
# 4. 主执行器 - 最终版，生成并保存为Hugging Face Datasets格式
# ===================================================================
def main():
    # --- 参数配置 ---
    num_samples = 10000
    val_ratio = 0.05
    test_ratio = 0.05
    output_dir = "datasets" # 输出目录名
    pruning_level = 1
    # ------------------

    print("--- Hugging Face 数据集生成脚本 ---")
    print(f"目标样本数: {num_samples}, 验证集比例: {val_ratio*100}%, 测试集比例: {test_ratio*100}%")

    # 步骤1: 顺序生成所有数据
    all_data = []
    for _ in tqdm(range(num_samples), desc="[1/3] 正在生成数据"):
        random_numbers = tuple(random.choices(range(1, 14), k=4)) # 数字范围扩大到13以增加难度
        all_data.append(generate_24_point_instance(random_numbers, pruning_level=pruning_level))

    # 步骤2: 随机打乱并精确分割
    print("\n[2/3] 正在分割数据集...")
    random.shuffle(all_data)

    num_total = len(all_data)
    num_test = int(num_total * test_ratio)
    num_val = int(num_total * val_ratio)

    test_data = all_data[:num_test]
    val_data = all_data[num_test : num_test + num_val]
    train_data = all_data[num_test + num_val :]
    
    print(f"分割结果: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")

    # 步骤3: 转换为Hugging Face Datasets对象
    train_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)
    test_dataset = datasets.Dataset.from_list(test_data)
    
    # 将三个split组合成一个DatasetDict
    dataset_dict = datasets.DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    print(f"\n[3/3] 正在保存数据集到 ./{output_dir} ...")
    
    # 步骤4: 保存到磁盘
    # 创建目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dataset_dict.save_to_disk(output_dir)

    print("\n[成功] 数据集已成功创建并保存在Hugging Face Datasets格式中！")
    print(f"您现在可以在训练脚本中使用 `datasets.load_from_disk('{output_dir}')` 来加载数据。")

if __name__ == "__main__":
    main()