import itertools
import random
import os
import re
from fractions import Fraction
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import json

# --- 配置区 ---
TOTAL_SAMPLES = 2000
# 调整比例：75%整数解，5%小数解，20%无解
TARGET_COUNTS = {
    'integer': int(TOTAL_SAMPLES * 0.75),    # 1500条
    'fractional': int(TOTAL_SAMPLES * 0.05), # 100条
    'unsolvable': int(TOTAL_SAMPLES * 0.20), # 400条
}

TARGET_COUNTS = {
    'integer': 1450,    # 1500条
    'fractional': 0, # 100条
    'unsolvable': 450, # 400条
}

# 预设一些已知会产生小数解的数字组合，提升小数解生成效率
KNOWN_FRACTIONAL_COMBINATIONS = [
    [1, 1, 8, 8],  # 8/(1-1/8)) = 64/7 ≈ 9.14 (需要重新计算)
    [3, 3, 8, 8],  # 8*3 = 24 (这个其实是整数解)
    [1, 2, 3, 6],  # 6/(1-2/3) = 18 (重新设计)
    [2, 3, 3, 4],  # 有小数解可能
    [1, 4, 5, 6],  # 6*5-1*4 = 26 (重新设计)
    [3, 4, 6, 8],  # 复杂组合
    [2, 5, 5, 6],  # 可能的小数解
    [1, 3, 5, 7],  # 测试组合
]

# --- 核心函数区 ---
# 定义运算符反操作映射
op_inv = {'+': '-', '-': '+', '*': '/', '/': '*'}

def find_first_solution(numbers):
    """
    基础解题器，用于快速找到任意一个解或确定无解。
    返回表达式字符串，如果无解则返回None。
    """
    ops = ['+', '-', '*', '/']
    for nums_perm in set(itertools.permutations(numbers)):
        for ops_perm in itertools.product(ops, repeat=3):
            expr1 = f"(({nums_perm[0]}{ops_perm[0]}{nums_perm[1]}){ops_perm[1]}{nums_perm[2]}){ops_perm[2]}{nums_perm[3]}"
            expr2 = f"({nums_perm[0]}{ops_perm[0]}{nums_perm[1]}){ops_perm[1]}({nums_perm[2]}{ops_perm[2]}{nums_perm[3]})"
            for expr in [expr1, expr2]:
                try:
                    if abs(eval(expr) - 24) < 1e-6:
                        return expr
                except (ZeroDivisionError, OverflowError):
                    continue
    return None

def find_all_solutions(numbers, max_solutions=3):
    """
    找到多个解法，用于增加解法多样性
    """
    ops = ['+', '-', '*', '/']
    solutions = []
    
    for nums_perm in set(itertools.permutations(numbers)):
        if len(solutions) >= max_solutions:
            break
        for ops_perm in itertools.product(ops, repeat=3):
            expr1 = f"(({nums_perm[0]}{ops_perm[0]}{nums_perm[1]}){ops_perm[1]}{nums_perm[2]}){ops_perm[2]}{nums_perm[3]}"
            expr2 = f"({nums_perm[0]}{ops_perm[0]}{nums_perm[1]}){ops_perm[1]}({nums_perm[2]}{ops_perm[2]}{nums_perm[3]})"
            for expr in [expr1, expr2]:
                try:
                    if abs(eval(expr) - 24) < 1e-6:
                        if expr not in solutions:
                            solutions.append(expr)
                            if len(solutions) >= max_solutions:
                                break
                except (ZeroDivisionError, OverflowError):
                    continue
    return solutions

def generate_solvable_puzzle(force_fractional=False):
    """
    改进的生成策略，对小数解使用预设组合提升效率
    """
    if force_fractional:
        # 80%几率使用预设组合，并验证是否真的产生小数解
        if random.random() < 0.8 and KNOWN_FRACTIONAL_COMBINATIONS:
            for _ in range(10):  # 尝试多个预设组合
                base_combo = random.choice(KNOWN_FRACTIONAL_COMBINATIONS)
                # 稍微变化数字，增加多样性
                numbers = []
                for num in base_combo:
                    if random.random() < 0.2:  # 20%几率微调
                        numbers.append(max(1, min(13, num + random.randint(-1, 1))))
                    else:
                        numbers.append(num)
                
                # 寻找所有解法，看是否有小数解
                all_solutions = find_all_solutions(numbers, max_solutions=5)
                for solution in all_solutions:
                    # 检查是否确实是小数解
                    is_fractional = False
                    if "/" in solution:
                        try:
                            result = eval(solution)
                            is_fractional = abs(result - round(result)) > 1e-9
                        except:
                            continue
                    
                    if is_fractional:
                        return numbers, solution
        
        # 如果预设组合失败，使用更激进的随机生成策略
        for _ in range(500):  # 增加尝试次数
            # 倾向于生成包含较小质数的组合，这样更容易产生除不尽的情况
            numbers = []
            for _ in range(4):
                if random.random() < 0.4:
                    numbers.append(random.choice([3, 5, 7, 11]))  # 质数
                else:
                    numbers.append(random.randint(1, 13))
            
            # 寻找所有解法
            all_solutions = find_all_solutions(numbers, max_solutions=10)
            for solution in all_solutions:
                is_fractional = False
                if "/" in solution:
                    try:
                        result = eval(solution)
                        is_fractional = abs(result - round(result)) > 1e-9
                    except:
                        continue
                
                if is_fractional:
                    return numbers, solution
    
    # 原有的随机生成逻辑（用于整数解）
    max_attempts = 100
    for _ in range(max_attempts):
        numbers = [random.randint(1, 13) for _ in range(4)]
        solution = find_first_solution(numbers)
        
        if solution:
            is_fractional = False
            if "/" in solution:
                try:
                    result = eval(solution)
                    is_fractional = abs(result - round(result)) > 1e-9
                except:
                    is_fractional = False
            
            if force_fractional and not is_fractional:
                continue
                
            return numbers, solution
    
    return None, None

def generate_enhanced_thought_process(numbers, solution=None):
    """
    生成更丰富的思维链，包含错误尝试、策略选择等
    """
    thought_log = []
    intro = f"我需要用数字 {', '.join(map(str, numbers))} 来计算24点游戏。\n"
    intro += "规则：每个数字用且仅用一次，通过加减乘除和括号得到24。\n\n"
    thought_log.append(intro)

    if solution:
        # 有解情况：展示思考过程
        strategy_thoughts = [
            f"让我先观察这些数字：{numbers}。我注意到其中有些数字的特点...\n",
            f"我想尝试几种常见策略：\n1. 找到能组成24的因子组合\n2. 先算出中间结果再组合\n3. 利用大数去调整小数\n\n"
        ]
        
        # 随机选择一些"错误尝试"来展示思考过程
        wrong_attempts = [
            f"首先试试 ({numbers[0]} + {numbers[1]}) * ({numbers[2]} + {numbers[3]}) = {(numbers[0]+numbers[1])*(numbers[2]+numbers[3])}，不对，不是24。\n",
            f"那试试 {numbers[0]} * {numbers[1]} * {numbers[2]} / {numbers[3]} = {numbers[0]*numbers[1]*numbers[2]/numbers[3] if numbers[3] != 0 else '无穷大'}，也不行。\n",
            f"再试试 ({numbers[0]} * {numbers[1]}) + ({numbers[2]} * {numbers[3]}) = {numbers[0]*numbers[1] + numbers[2]*numbers[3]}，还是不对。\n"
        ]
        
        # 随机选择1-2个错误尝试
        selected_attempts = random.sample(wrong_attempts, min(2, len(wrong_attempts)))
        
        thought_log.extend(strategy_thoughts)
        thought_log.extend(selected_attempts)
        
        # 成功解法
        success_part = f"经过多次尝试，我发现了一个可行的表达式：{solution}\n"
        try:
            result = eval(solution)
            success_part += f"让我验证一下：{solution} = {result:.6f}\n"
            if abs(result - 24) < 1e-6:
                success_part += "太好了！结果确实是24，这就是正确答案。\n"
            else:
                success_part += f"咦，计算结果是{result:.6f}，让我重新检查...\n"
        except:
            success_part += "让我仔细验证这个表达式的正确性...\n"
        
        thought_log.append(success_part)
        
        # 如果有多个解法，提及其他可能性
        other_solutions = find_all_solutions(numbers, max_solutions=2)
        if len(other_solutions) > 1:
            thought_log.append(f"顺便提一下，这组数字还有其他解法，比如 {other_solutions[1] if len(other_solutions) > 1 else '其他表达式'}。\n")
            
    else:
        # 无解情况：展示系统性搜索过程
        search_attempts = [
            f"我先尝试所有加法组合：\n- ({numbers[0]}+{numbers[1]}) 和 ({numbers[2]}+{numbers[3]}) 的各种运算\n- ({numbers[0]}+{numbers[2]}) 和 ({numbers[1]}+{numbers[3]}) 的各种运算\n都无法得到24。\n\n",
            f"然后尝试乘法为主的组合：\n- {numbers[0]}*{numbers[1]} = {numbers[0]*numbers[1]}，然后与{numbers[2]}, {numbers[3]}组合\n- {numbers[2]}*{numbers[3]} = {numbers[2]*numbers[3]}，然后与{numbers[0]}, {numbers[1]}组合\n依然无法得到24。\n\n",
            f"最后尝试除法和混合运算：\n- 各种包含除法的表达式\n- 复杂的括号组合\n经过穷尽搜索，确认无解。\n"
        ]
        
        thought_log.extend(search_attempts)
        thought_log.append("经过系统性的验证，我确信这组数字无法通过四则运算得到24。\n")

    final_thought = "<think>\n" + "".join(thought_log) + "</think>"
    return final_thought

def format_item(numbers, solution, thought):
    """统一格式化输出"""
    final_answer = solution if solution else "无解"
    system_prompt = "你是一个精通数学的智能助手，擅长通过自然语言推理来解决问题。你的任务是解决24点游戏。在给出最终答案前，请务必先在<think>标签中展示你详细的、人性化的思考过程，包括尝试、失败、重新思考等真实的推理步骤。如果问题无解，请明确说明。"
    user_prompt = f"请为以下这组数字计算24点：{', '.join(map(str, numbers))}"
    formatted_output = f"{thought}\n\n最终答案：{final_answer}"
    return {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": formatted_output}]}


# --- 主程序：三路并进的混合策略 ---

def create_final_dataset():
    datasets = {'integer': [], 'fractional': [], 'unsolvable': []}
    generated_sets = set()
    # 为整数解允许重复数字组合，但确保解法不同
    generated_integer_solutions = set()

    # 任务1: 生成无解案例
    print("任务1：正在生成无解案例...")
    pbar = tqdm(total=TARGET_COUNTS['unsolvable'], desc="寻找无解案例")
    while len(datasets['unsolvable']) < TARGET_COUNTS['unsolvable']:
        numbers = tuple(sorted([random.randint(1, 13) for _ in range(4)]))
        if numbers in generated_sets: continue
        
        if find_first_solution(list(numbers)) is None:
            generated_sets.add(numbers)
            thought = generate_enhanced_thought_process(list(numbers), solution=None)
            datasets['unsolvable'].append(format_item(list(numbers), None, thought))
            pbar.update(1)
    pbar.close()

    # 任务2: 生成整数解案例（修改策略）
    print("任务2：正在生成整数解案例...")
    pbar_int = tqdm(total=TARGET_COUNTS['integer'], desc="构造整数解案例")
    attempts_int = 0
    consecutive_failures = 0
    max_attempts_per_check = 50000  # 减少检查间隔
    
    while len(datasets['integer']) < TARGET_COUNTS['integer']:
        attempts_int += 1
        numbers, solution = generate_solvable_puzzle(force_fractional=False)
        
        if numbers:
            numbers_tuple = tuple(sorted(numbers))
            
            # 检查是否为整数解
            is_fractional_solution = False
            if "/" in solution:
                try:
                    result = eval(solution)
                    is_fractional_solution = abs(result - round(result)) > 1e-9
                except:
                    is_fractional_solution = False
            
            if not is_fractional_solution:
                # 使用解法作为唯一性标识，而不是数字组合
                solution_key = (numbers_tuple, solution)
                
                # 如果这个特定的解法还没有被使用过
                if solution_key not in generated_integer_solutions:
                    generated_integer_solutions.add(solution_key)
                    generated_sets.add(numbers_tuple)  # 仍然记录数字组合避免无解重复
                    thought = generate_enhanced_thought_process(numbers, solution)
                    datasets['integer'].append(format_item(numbers, solution, thought))
                    pbar_int.update(1)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
        else:
            consecutive_failures += 1
        
        # 更频繁的检查和更早的终止
        if consecutive_failures > max_attempts_per_check:
            print(f"\n警告：连续{consecutive_failures}次未找到新的整数解")
            print(f"当前已生成{len(datasets['integer'])}个整数解")
            
            # 计算成功率
            success_rate = len(datasets['integer']) / attempts_int if attempts_int > 0 else 0
            print(f"当前成功率: {success_rate:.6f}")
            
            if success_rate < 0.000001:  # 成功率极低，认为已达到理论上限
                print("成功率过低，认为已接近理论上限")
                break
            
            # 动态调整目标数量，但设置最小值
            current_count = len(datasets['integer'])
            if current_count >= 1400:  # 如果已经有1400个，就接受当前数量
                print(f"已生成{current_count}个整数解，接近目标，停止生成")
                TARGET_COUNTS['integer'] = current_count
                break
            
            consecutive_failures = 0
            max_attempts_per_check = min(max_attempts_per_check * 2, 100000)  # 增加检查间隔
        
        if attempts_int > 0 and attempts_int % 50000 == 0:
            print(f"\n整数解生成已尝试{attempts_int}次，当前已生成{len(datasets['integer'])}个")
            print(f"去重解法数：{len(generated_integer_solutions)}")
            success_rate = len(datasets['integer']) / attempts_int
            print(f"成功率: {success_rate:.6f}")
    
    pbar_int.close()
    print(f"整数解生成完成，总尝试次数: {attempts_int}")
    print(f"实际生成的整数解数量: {len(datasets['integer'])}")
    
    # 如果整数解不够，从其他类型补充
    total_needed = TOTAL_SAMPLES
    current_total = len(datasets['integer']) + len(datasets['fractional']) + len(datasets['unsolvable'])
    
    if len(datasets['integer']) < TARGET_COUNTS['integer']:
        shortage = TARGET_COUNTS['integer'] - len(datasets['integer'])
        print(f"整数解不足{shortage}个，将调整其他类型比例来保持总数")
        
        # 调整其他类型的目标数量来补偿
        remaining_samples = total_needed - len(datasets['integer'])
        TARGET_COUNTS['fractional'] = min(TARGET_COUNTS['fractional'], remaining_samples // 10)  # 最多10%
        TARGET_COUNTS['unsolvable'] = remaining_samples - TARGET_COUNTS['fractional']
        
        print(f"调整后目标: 整数解{len(datasets['integer'])}, 小数解{TARGET_COUNTS['fractional']}, 无解{TARGET_COUNTS['unsolvable']}")

    # 任务3: 生成小数解案例（减少目标数量，提升效率）
    print("任务3：正在生成小数解案例...")
    pbar_frac = tqdm(total=TARGET_COUNTS['fractional'], desc="构造小数解案例")
    attempts_frac = 0
    consecutive_failures_frac = 0
    
    while len(datasets['fractional']) < TARGET_COUNTS['fractional']:
        attempts_frac += 1
        numbers, solution = generate_solvable_puzzle(force_fractional=True)
        
        if numbers:
            numbers_tuple = tuple(sorted(numbers))
            if numbers_tuple in generated_sets: 
                consecutive_failures_frac += 1
                continue
            
            is_fractional_solution = False
            if "/" in solution:
                try:
                    result = eval(solution)
                    is_fractional_solution = abs(result - round(result)) > 1e-9
                except:
                    is_fractional_solution = False
            
            if is_fractional_solution:
                generated_sets.add(numbers_tuple)
                thought = generate_enhanced_thought_process(numbers, solution)
                datasets['fractional'].append(format_item(numbers, solution, thought))
                pbar_frac.update(1)
                consecutive_failures_frac = 0
                print(f"\n找到小数解: {numbers} -> {solution}")
            else:
                consecutive_failures_frac += 1
        else:
            consecutive_failures_frac += 1
        
        # 如果连续失败太多次，降低目标或终止
        if consecutive_failures_frac > 5000:
            print(f"\n小数解生成困难，连续{consecutive_failures_frac}次失败")
            print(f"当前已生成{len(datasets['fractional'])}个小数解")
            
            if len(datasets['fractional']) >= 50:  # 至少生成50个
                print("降低小数解目标数量")
                TARGET_COUNTS['fractional'] = len(datasets['fractional'])
                break
            
            consecutive_failures_frac = 0
        
        if attempts_frac > 0 and attempts_frac % 1000 == 0:
            print(f"\n小数解生成已尝试{attempts_frac}次，当前已生成{len(datasets['fractional'])}个")
    
    pbar_frac.close()
    print(f"小数解生成完成，总尝试次数: {attempts_frac}")

    # --- 数据整合 ---
    print(f"\n最终数据统计:")
    print(f"整数解: {len(datasets['integer'])}")
    print(f"小数解: {len(datasets['fractional'])}")
    print(f"无解: {len(datasets['unsolvable'])}")
    
    final_data = datasets['integer'] + datasets['fractional'] + datasets['unsolvable']
    random.shuffle(final_data)
    return final_data

if __name__ == '__main__':
    all_data = create_final_dataset()
    print(f"\n成功生成 {len(all_data)} 条混合类型数据。")
    
    random.shuffle(all_data)
    val_test_size = int(0.05 * len(all_data))
    test_data = all_data[:val_test_size]
    validation_data = all_data[val_test_size : 2 * val_test_size]
    train_data = all_data[2 * val_test_size :]

    print("\n数据切分完成:")
    print(f"训练集数量: {len(train_data)}")
    print(f"验证集数量: {len(validation_data)}")
    print(f"测试集数量: {len(test_data)}")

    game_24_dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(validation_data),
        'test': Dataset.from_list(test_data)
    })

    save_directory = "datasets"
    print(f"\n正在将数据集保存到 ./{save_directory} ...")
    try:
        game_24_dataset_dict.save_to_disk(save_directory)
        print(f"数据集已成功保存！请查看 '{os.path.abspath(save_directory)}' 文件夹。")
    except Exception as e:
        print(f"保存失败：{e}")