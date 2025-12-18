# 导入必要的标准库和第三方库
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file

# 定义 Hugging Face 模型参数名到目标推理框架参数名的映射表
# 格式: "HF 参数名片段": ("目标参数名", 切分维度)
# - 切分维度为 None 表示该张量不进行模型并行切分（如 LayerNorm、bias 等）
# - 切分维度为 0 或 1 表示沿该维度进行切分（0=输出通道，1=输入通道）
mapping = {
    "embed_tokens": ("embed", 0),                    # 词嵌入层，按 vocab 维度切分
    "input_layernorm": ("attn_norm", None),          # Attention 前的归一化层，不切分
    "post_attention_layernorm": ("ffn_norm", None),  # FFN 前的归一化层，不切分
    "q_proj": ("wq", 0),                             # Q 投影矩阵，按输出通道切分（dim=0）
    "q_a_proj": ("wq_a", None),                      # Q 的低秩投影 A（如 DeepSeek-MoE），不切分
    "q_a_layernorm": ("q_norm", None),               # Q_A 后的归一化，不切分
    "q_b_proj": ("wq_b", 0),                         # Q 的低秩投影 B，按输出通道切分
    "kv_a_proj_with_mqa": ("wkv_a", None),           # KV 的低秩投影（MQA 架构），不切分
    "kv_a_layernorm": ("kv_norm", None),             # KV_A 后的归一化，不切分
    "kv_b_proj": ("wkv_b", 0),                       # KV 的低秩投影 B，按输出通道切分
    "o_proj": ("wo", 1),                             # Attention 输出投影，按输入通道切分（dim=1）
    "gate": ("gate", None),                          # MoE 门控网络权重，通常不切分
    "gate_proj": ("w1", 0),                          # FFN 中 SwiGLU 的 gate 投影，按输出切分
    "down_proj": ("w2", 1),                          # FFN 输出投影（SwiGLU 后），按输入切分
    "up_proj": ("w3", 0),                            # FFN 中 SwiGLU 的 up 投影，按输出切分
    "norm": ("norm", None),                          # 最终归一化层（如 RMSNorm），不切分
    "lm_head": ("head", 0),                          # 语言模型头，按 vocab 维度切分
    "scale": ("scale", None),                        # 量化相关的 scale 参数，不切分
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    # 设置 PyTorch CPU 加载线程数，避免加载时占用过多系统资源
    torch.set_num_threads(8)

    # 计算每个模型并行 rank（即每个 GPU）应分配的本地专家数量
    n_local_experts = n_experts // mp

    # 为每个 MP rank 初始化一个空字典，用于存储该 rank 所需的参数子集
    state_dicts = [{} for _ in range(mp)]

    # 遍历 HF 检查点目录下所有 .safetensors 文件（通常为分片存储）
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        # 使用 safe_open 安全地打开 safetensors 文件（延迟加载，节省内存）
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # 遍历当前文件中所有参数名
            for name in f.keys():
                # 【硬编码跳过】第61层（可能是特定模型的异常层，如 Qwen-MoE 的临时 workaround）
                if "model.layers.61" in name:
                    continue

                # 从 safetensors 中实际读取张量数据（此时才加载到内存）
                param: torch.Tensor = f.get_tensor(name)

                # 移除 Hugging Face 模型参数名中的 "model." 前缀
                if name.startswith("model."):
                    name = name[len("model."):]

                # 将模块名称标准化为目标推理框架的命名规范
                name = name.replace("self_attn", "attn")  # 自注意力 → attn
                name = name.replace("mlp", "ffn")  # MLP → FFN
                name = name.replace("weight_scale_inv", "scale")  # 量化 scale 重命名
                name = name.replace("e_score_correction_bias", "bias")  # bias 字段重命名

                # 提取参数名中倒数第二段作为 key（通常是层类型，如 q_proj, down_proj）
                key = name.split(".")[-2]
                # 确保该 key 在预定义的映射表中，否则报错
                assert key in mapping, f"Key {key} not found in mapping"

                # 获取目标参数名和切分维度
                new_key, dim = mapping[key]
                # 将原 key 替换为目标 key
                name = name.replace(key, new_key)

                # 对每个模型并行 rank（0 到 mp-1）进行参数分配
                for i in range(mp):
                    new_param = param  # 默认使用完整参数

                    # 情况1：当前参数属于 MoE 专家层（且不是 shared_experts）
                    if "experts" in name and "shared_experts" not in name:
                        # 从参数名中提取 expert 的索引（假设格式为 ...experts.{idx}.xxx...）
                        idx = int(name.split(".")[-3])
                        # 如果该 expert 不在当前 rank i 的分配范围内，则跳过（不加入该 rank 的 state dict）
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue

                    # 情况2：非专家层，但需要沿指定维度进行模型并行切分
                    elif dim is not None:
                        # 检查该维度是否可被 mp 整除（否则无法均匀切分）
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        # 计算每个 rank 分到的 shard 大小
                        shard_size = param.size(dim) // mp
                        # 沿维度 dim 切片（narrow 返回 view，contiguous 确保内存连续）
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()

                    # 将处理后的参数存入对应 rank 的字典中
                    state_dicts[i][name] = new_param

    # 创建输出目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)

    # 将每个 rank 的参数字典保存为独立的 safetensors 文件
    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # 复制 tokenizer 相关文件（如 tokenizer.json, tokenizer.model, special_tokens_map.json 等）
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


# 主程序入口：解析命令行参数并执行转换
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True, help="Path to Hugging Face checkpoint directory")
    parser.add_argument("--save-path", type=str, required=True, help="Output directory for converted checkpoints")
    parser.add_argument("--n-experts", type=int, required=True, help="Total number of experts in the MoE model")
    parser.add_argument("--model-parallel", type=int, required=True, help="Model parallelism factor (number of shards)")
    args = parser.parse_args()

    # 校验：专家总数必须能被模型并行度整除，确保均匀分配
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"

    # 执行主转换逻辑
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)

