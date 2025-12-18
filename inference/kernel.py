# 类型提示和核心依赖
from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    # 获取当前程序实例 ID（每个 block 对应一个 program）
    pid = tl.program_id(axis=0)
    # 计算当前 block 的全局偏移索引
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 从全局内存加载输入块（转为 float32 进行计算）
    x = tl.load(x_ptr + offs).to(tl.float32)
    # 计算该 block 的绝对值最大值（用于动态缩放）
    amax = tl.max(tl.abs(x))  # reduction 操作
    # 防止除零：将 amax 下限 clamp 到 1e-4
    amax = tl.maximum(amax, 1e-4)
    # 默认缩放因子：amax / 448（对应 float8_e4m3fn 的最大可表示值 ≈ 448）
    s = amax / 448.

    # 可选缩放格式：若为 "ue8m0"（无符号指数格式），则对 scale 取 2 的整数次幂（便于硬件实现）
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))  # 向上取整到最近的 2 的幂
        s = tl.math.exp2(exp)  # s = 2^exp

    # 量化：x / s，然后转换为目标 dtype（如 float8_e4m3fn）
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    # 存储量化后的值和缩放因子
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)  # 每个 block 一个 scale


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    # 输入校验：必须是连续张量，且最后一维能被 block_size 整除
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(
        -1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'

    # 初始化输出：量化结果（float8_e4m3fn）和 scale（float32）
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # scale 张量形状：除最后一维外相同，最后一维压缩为原长 / block_size
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    # 定义 grid：总元素数 / BLOCK_SIZE，向上取整
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    # 启动 Triton kernel
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    # 二维 grid：(pid_m, pid_n) 对应输出矩阵的 tile
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # 每行方向有多少个 block（用于计算 scale 索引）
    n = tl.cdiv(N, BLOCK_SIZE)

    # 当前 tile 的行/列偏移
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 展平为一维索引（行优先）
    offs = offs_m[:, None] * N + offs_n[None, :]
    # 边界掩码（防止越界）
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 加载量化权重并转为 float32
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # 加载对应的 scale：假设 scale 是按 (M//BS, N//BS) 分块存储
    s = tl.load(s_ptr + pid_m * n + pid_n)
    # 反量化：x * s
    y = x * s
    # 存储结果
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    # 校验输入
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'

    M, N = x.size()
    # 输出张量使用默认 dtype（如 bfloat16 或 float16）
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    # 二维 grid：(M/BLOCK, N/BLOCK)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# 定义 FP8 GEMM 的自动调优配置空间
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    # 获取当前 tile 的坐标
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # K 方向需要迭代的次数
    k = tl.cdiv(K, BLOCK_SIZE_K)

    # 当前 tile 的 M/N 偏移（带模防止越界）
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 指针初始化：A 按行主序，B 按列主序（实际 B 被转置存储）
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    # Scale 指针：
    # - A 的 scale 按行分块（每 BLOCK_SIZE_K 列一组），所以每行有 k 个 scale
    a_s_ptrs = a_s_ptr + offs_m * k
    # - B 的 scale 按列分块（每 BLOCK_SIZE_K 行一组），所以每列有 k 个 scale
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    # 初始化累加器（高精度 float32）
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K 方向分块循环
    for i in range(k):
        # 加载 A 和 B 的当前 K 块（带边界掩码）
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)

        # 加载对应的 scale
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)

        # 执行点积，并应用 scale：C += (A * a_s) @ (B * b_s) = (A @ B) * a_s * b_s
        # 注意：a_s 是 (BLOCK_M,)，b_s 是 (BLOCK_N,)，广播后相乘
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        # 移动指针到下一个 K 块
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1

    # 将结果转为目标输出 dtype（如 bfloat16）
    c = accumulator.to(c_ptr.dtype.element_ty)

    # 存储最终结果（带边界掩码）
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    # 输入校验
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'

    # 推导矩阵维度：支持 batched GEMM（a 最后一维为 K）
    K = a.size(-1)
    M = a.numel() // K  # 总行数（展平 batch 维度）
    N = b.size(0)  # B 的行数即 C 的列数（注意：此处假设 B 是 (N, K)，即已转置）

    # 初始化输出张量
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    # 启动自动调优的 GEMM kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c

