import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from swanlab.integration.transformers import SwanLabCallback


def main():
    """主训练函数"""

    # --- 1. 配置与超参数定义 ---

    # 模型与分词器配置
    model_id = "model/Qwen3-0.6B"  # 使用我们选择的Qwen3-0.6B模型

    # 数据集配置
    dataset_path = "./datasets"  # 数据集保存的目录

    swanlab_callback = SwanLabCallback(
        project="twenty-four-point-game",
        experiment_name="Qwen3-0.6B-Long-CoT-SFT",
        description="Qwen3-0.6B模型在24点游戏长CoT数据集上的SFT训练",
    )

    # 训练参数配置 (TrainingArguments)
    # 这里定义了所有与训练过程相关的超参数
    training_args = SFTConfig(
        output_dir="./results",  # 训练输出物（如checkpoint）的保存目录
        num_train_epochs=3,  # 训练轮次
        per_device_train_batch_size=4,  # 每块GPU上的批次大小。对于3090和0.6B模型，4或8是安全的选择
        gradient_accumulation_steps=4,  # 梯度累积步数。有效批次大小 = batch_size * num_gpus * accumulation_steps
        gradient_checkpointing=False,  # 梯度检查点，用计算时间换取显存
        learning_rate=1e-5,  # 学习率
        lr_scheduler_type="cosine",  # 学习率调度器类型
        warmup_ratio=0.03,  # 预热步数比例
        logging_steps=20,  # 每隔多少步记录一次日志
        save_strategy="epoch",  # 模型保存策略（每轮保存一次）
        bf16=True,  # 为RTX 3090启用BF16混合精度训练
        # --- 集成与分布式训练配置 ---
        report_to="none",  # 将日志报告给SwanLab
        deepspeed="deepspeed_zero2.json",  # DeepSpeed配置文件路径
        dataset_text_field="messages",  # 指定数据集中包含对话内容的字段名
        max_length=2 * 1024,  # 序列最大长度。根据您的CoT长度和显存进行调整
        packing=False,  # Packing是一种将多个短序列打包成一个长序列的技术，对于我们的长CoT数据，应关闭
    )

    # --- 2. 加载模型、分词器和数据集 ---

    print("--- [1/4] 正在加载模型和分词器 ---")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Qwen模型通常没有默认的pad_token，我们将其设置为eos_token以避免警告
    tokenizer.pad_token = tokenizer.eos_token

    # 加载4-bit量化后的模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": int(os.environ.get("LOCAL_RANK"))},  # 将模型加载到当前GPU
        # device_map="auto",  # 自动选择设备映射
        # device_map="cuda",  # 将模型加载到所有可用的GPU上
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 使用bfloat16以节省显存
        attn_implementation="flash_attention_2",  # 使用Flash Attention以提升速度和显存效率
    )

    print("--- [2/4] 正在加载并准备数据集 ---")

    # 从磁盘加载我们之前创建的数据集
    try:
        dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        print(f"[错误] 数据集未在 '{dataset_path}' 找到。请先运行数据生成脚本。")
        return

    print("数据集加载成功:")
    print(dataset)

    # --- 3. 初始化SFTTrainer ---

    print("--- [3/4] 正在初始化SFTTrainer ---")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],  # 提供验证集以在训练过程中评估性能
        callbacks=[swanlab_callback],  # 集成SwanLab回调以进行实验跟踪
    )

    # --- 4. 开始训练与保存 ---

    print("--- [4/4] 开始训练 ---")

    # 启动训练
    train_result = trainer.train()

    print("--- 训练完成 ---")

    # 保存最终的LoRA适配器
    # 最终的模型权重会保存在 training_args.output_dir 下的 "final_checkpoint" 子目录
    final_checkpoint_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)

    print(f"训练统计: {train_result.metrics}")
    print(f"最终的LoRA适配器已成功保存到: {final_checkpoint_dir}")
    print("您现在可以使用该适配器进行推理和评估。")


if __name__ == "__main__":
    main()
