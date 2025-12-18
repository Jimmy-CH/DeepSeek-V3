# 导入所需模块
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

# 从自定义 kernel 模块导入反量化函数（用于将 FP8 权重还原为 BF16）
from kernel import weight_dequant


def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    # 设置 PyTorch 默认张量类型为 bfloat16（BF16），确保新权重以 BF16 格式存储
    torch.set_default_dtype(torch.bfloat16)

    # 创建输出目录（如果不存在）
    os.makedirs(bf16_path, exist_ok=True)

    # 读取原始模型的索引文件（记录每个参数属于哪个 .safetensors 文件）
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]  # 参数名 -> 文件名 的映射

    # 缓存已加载的 safetensors 文件内容，避免重复读盘（优化 I/O 和显存）
    loaded_files = {}
    # 记录所有被识别为 FP8 的权重名称（用于后续更新索引文件）
    fp8_weight_names = []

    # 定义辅助函数：根据参数名从正确的文件中获取张量（支持跨文件查找 scale_inv）
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        # 根据 weight_map 找到该 tensor 所在的文件名
        file_name = weight_map[tensor_name]
        # 如果该文件尚未加载，则从磁盘加载并缓存到 GPU 显存
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        # 返回对应 tensor
        return loaded_files[file_name][tensor_name]

    # 获取所有 .safetensors 文件路径，并排序以保证处理顺序一致
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()

    # 遍历每个 safetensors 文件进行转换
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        # 加载当前文件的所有张量到 GPU
        current_state_dict = load_file(safetensor_file, device="cuda")
        # 将其加入缓存（便于 get_tensor 函数访问）
        loaded_files[file_name] = current_state_dict

        # 构建新的 state dict（用于保存 BF16 权重）
        new_state_dict = {}

        # 遍历当前文件中的每个参数
        for weight_name, weight in current_state_dict.items():
            # 跳过 _scale_inv 张量（这些是量化辅助参数，不需要保留）
            if weight_name.endswith("_scale_inv"):
                continue
            # 判断是否为 FP8 权重：FP8 张量的 element_size() == 1 字节
            elif weight.element_size() == 1:  # FP8 weight
                # 构造对应的 scale_inv 参数名（如 "model.layers.0.mlp.down_proj.weight_scale_inv"）
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # 通过 get_tensor 从任意文件中获取 scale_inv（可能不在当前文件！）
                    scale_inv = get_tensor(scale_inv_name)
                    # 记录该权重为 FP8 类型（用于后续清理索引）
                    fp8_weight_names.append(weight_name)
                    # 调用自定义反量化函数，将 FP8 + scale_inv 还原为 BF16
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    # 如果找不到对应的 scale_inv，则跳过反量化（保留原 FP8，但通常不应发生）
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                # 非 FP8 权重（如 LayerNorm bias、embedding 等）直接保留
                new_state_dict[weight_name] = weight

        # 将转换后的 BF16 权重保存到输出目录（文件名不变）
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # 内存管理：仅保留最近使用的 2 个文件缓存，防止显存溢出
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))  # 获取最早加入的缓存
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()  # 主动释放 GPU 显存

    # 更新 model.safetensors.index.json：移除所有 *_scale_inv 的条目
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)  # 删除 scale_inv 的映射
    # 保存新的索引文件（metadata 保留为空字典）
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


# 主程序入口：解析命令行参数并调用 main 函数
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Path to HF-style FP8 model directory")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="Output path for converted BF16 model")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
