# 导入标准库和深度学习相关模块
import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

# 导入自定义模型类和配置参数结构
from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    # 防止除零：temperature 至少为 1e-5
    logits = logits / max(temperature, 1e-5)
    # 应用 softmax 得到概率分布
    probs = torch.softmax(logits, dim=-1)
    # 使用 Gumbel-Max 技巧进行采样（等价于多项式采样）
    # 公式：argmax(log(p) - log(-log(u))) ≈ argmax(p / (-log(u)))
    # 此处简化为 p / Exp(1)，再取 argmax
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()  # 启用推理模式，禁用梯度计算以提升性能
def generate(
        model: Transformer,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    # 获取每个 prompt 的长度
    prompt_lens = [len(t) for t in prompt_tokens]
    # 确保所有 prompt 不超过模型最大上下文长度
    assert max(
        prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"

    # 计算实际生成总长度（不超过模型最大长度）
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))

    # 初始化 token 张量，用 -1 填充（表示未生成位置）
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")

    # 将 prompt 填入 tokens 的起始位置
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0  # 上一次 forward 的结束位置（用于 KV Cache）
    # 标记每条序列是否已生成结束符
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    # 创建 mask：True 表示是 prompt 部分（不应被覆盖）
    prompt_mask = tokens != -1

    # 从 prompt 结束位置开始逐个生成新 token
    for cur_pos in range(min(prompt_lens), total_len):
        # 模型前向传播：仅传入新增的 token（利用 KV Cache）
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

        # 根据 temperature 决定采样方式
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)  # greedy decoding

        # 如果当前位置属于 prompt（如 batch 中各序列长度不同），保留原 token
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        # 更新 finished：当非 prompt 位置生成了 eos_id 时标记完成
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos

        # 若所有序列都已完成，提前退出
        if finished.all():
            break

    # 提取生成结果（去除 prompt 部分，并截断至 max_new_tokens 或 eos）
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # 只取生成部分，最多 max_new_tokens 个
        toks = toks[prompt_lens[i]:prompt_lens[i] + max_new_tokens]
        # 若包含 eos，则截断到 eos 之前
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
        ckpt_path: str,
        config: str,
        input_file: str = "",
        interactive: bool = True,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    # 获取分布式训练环境变量（由 torchrun 设置）
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # 若为多卡，初始化 NCCL 分布式后端
    if world_size > 1:
        dist.init_process_group("nccl")

    # 仅 rank 0 打印日志，其他 rank 屏蔽 print（避免重复输出）
    global print
    if rank != 0:
        print = lambda *_, **__: None

    # 绑定当前进程到对应 GPU
    torch.cuda.set_device(local_rank)
    # 设置默认张量类型为 bfloat16（与模型权重一致）
    torch.set_default_dtype(torch.bfloat16)
    # 限制 CPU 线程数
    torch.set_num_threads(8)
    # 固定随机种子（确保采样可复现）
    torch.manual_seed(965)

    # 加载模型配置
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    # 在 CUDA 设备上下文中实例化模型（避免主机内存占用）
    with torch.device("cuda"):
        model = Transformer(args)

    # 加载 tokenizer（从 HF 格式目录）
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # 【预热】执行一次 dummy forward（可能用于 CUDA 图编译或缓存分配）
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])

    # 加载当前 rank 对应的模型权重分片（文件名格式：model{rank}-mp{world_size}.safetensors）
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    # 交互式生成模式
    if interactive:
        messages = []  # 存储对话历史（符合 chat template 格式）
        while True:
            # 仅 rank 0 接收用户输入，并广播给其他 rank（保证所有 GPU 输入一致）
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]

            # 特殊命令处理
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue

            # 构建对话消息并应用 tokenizer 的 chat template
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            # 生成回复
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)

            # 仅 rank 0 打印结果
            print(completion)

            # 将模型回复加入对话历史
            messages.append({"role": "assistant", "content": completion})

    # 批处理模式：从文件读取多行 prompt
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        # 检查 batch size 是否超出模型限制
        assert len(
            prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"

        # 为每个 prompt 构建单轮对话模板
        prompt_tokens = [
            tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
            for prompt in prompts
        ]

        # 批量生成
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        # 打印结果（仅 rank 0）
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    # 若使用分布式，清理进程组
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    # 必须指定交互模式或输入文件之一
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
