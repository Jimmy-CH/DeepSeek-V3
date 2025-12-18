import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

# 导入自定义的 FP8 量化/反量化与 GEMM 内核（假设已实现）
from kernel import act_quant, weight_dequant, fp8_gemm

# 全局变量：分布式设置（默认单卡）
world_size = 1  # 总进程数（GPU 数）
rank = 0  # 当前进程编号
block_size = 128  # 量化块大小
gemm_impl: Literal["bf16", "fp8"] = "bf16"  # 矩阵乘法实现方式
attn_impl: Literal["naive", "absorb"] = "absorb"  # 注意力实现方式


@dataclass
class ModelArgs:
    """
    模型参数配置类，使用 dataclass 定义超参数。
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4  # 支持长上下文（16k）
    dtype: Literal["bf16", "fp8"] = "bf16"  # 计算数据类型
    scale_fmt: Optional[str] = None  # 量化比例格式（如 'e5m2'）
    vocab_size: int = 102400  # 词汇表大小
    dim: int = 2048  # 模型隐藏维度
    inter_dim: int = 10944  # Dense MLP 中间维度
    moe_inter_dim: int = 1408  # MoE 专家中间维度
    n_layers: int = 27  # 总层数
    n_dense_layers: int = 1  # 前 n_dense_layers 层使用 Dense MLP，其余用 MoE
    n_heads: int = 16  # 注意力头数

    # MoE 相关参数
    n_routed_experts: int = 64  # 路由专家总数
    n_shared_experts: int = 2  # 共享专家数（始终激活）
    n_activated_experts: int = 6  # 每 token 激活的专家数
    n_expert_groups: int = 1  # 专家分组数（用于分组路由）
    n_limited_groups: int = 1  # 限制激活的组数
    score_func: Literal["softmax", "sigmoid"] = "softmax"  # 路由打分函数
    route_scale: float = 1.  # 路由权重缩放因子

    # MLA (Multi-head Latent Attention) 参数
    q_lora_rank: int = 0  # Q 投影的 LoRA 秩（0 表示不用 LoRA）
    kv_lora_rank: int = 512  # KV 投影的低秩维度
    qk_nope_head_dim: int = 128  # QK 中无 RoPE 部分的维度
    qk_rope_head_dim: int = 64  # QK 中带 RoPE 部分的维度
    v_head_dim: int = 128  # Value 头维度

    # YaRN RoPE 扩展参数
    original_seq_len: int = 4096  # 原始训练序列长度
    rope_theta: float = 10000.0  # RoPE 基础频率
    rope_factor: float = 40  # 序列扩展因子
    beta_fast: int = 32  # 快速衰减旋转数
    beta_slow: int = 1  # 慢速衰减旋转数
    mscale: float = 1.  # 注意力缩放补偿因子


class ParallelEmbedding(nn.Module):
    """
    支持模型并行的词嵌入层：将词汇表按 world_size 切分，每个 GPU 存储一部分。
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # 确保词汇表可被 world_size 整除
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 只存储本 rank 负责的词向量子集
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        - 将输入 token 映射到本地词汇区间
        - 对超出范围的 token 设为 0（避免索引错误）
        - 使用 all_reduce 聚合所有 GPU 的嵌入结果
        """
        if world_size > 1:
            # 创建掩码：标记不在本 rank 词汇区间的 token
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx  # 转为本地索引
            x[mask] = 0  # 安全处理越界索引
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0  # 越界位置输出置零
            dist.all_reduce(y)  # 聚合所有 GPU 的嵌入结果
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    自定义线性层，支持 FP8 量化推理。
    根据 weight 的 element_size 判断是否量化（1 字节 = FP8）。
    """
    if weight.element_size() > 1:
        # 非量化权重（如 bf16），直接调用标准线性层
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # FP8 权重但使用 bf16 GEMM：先反量化再计算
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # FP8 GEMM 路径：对激活量化，调用自定义 FP8 GEMM
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    支持量化权重的线性层基类。
    若权重为 FP8（element_size=1），则自动注册 scale 参数。
    """
    dtype = torch.bfloat16  # 默认数据类型
    scale_fmt: Optional[str] = None  # 量化比例格式

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重（可能为 FP8）
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            # FP8 权重：计算 scale 张量形状（按 block_size 分块）
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        # 偏置项
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    列并行线性层：将输出维度（out_features）按 world_size 切分。
    每个 GPU 计算部分输出，无需通信（除非后续需要聚合）。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    行并行线性层：将输入维度（in_features）按 world_size 切分。
    每个 GPU 计算部分结果，需 all_reduce 聚合。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)  # 聚合各 GPU 的部分结果
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    RMS 归一化层：替代 LayerNorm，无偏置，仅缩放。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算 RoPE（旋转位置编码）所需的复数频率张量。
    支持 YaRN 方法进行长序列外推。
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """计算修正维度（YaRN 核心）"""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """计算修正维度范围"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """线性平滑函数，用于混合原始与修正频率"""
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 基础频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        # YaRN 修正：对高频部分进行缩放和平滑
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 生成位置索引
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    # 转为复数形式：cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码（RoPE）。
    将输入 x 视为复数，与 freqs_cis 相乘后转回实数。
    """
    dtype = x.dtype
    # 将最后维度 reshape 为 (..., dim//2, 2)，再转为复数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) 层，结合 LoRA 和 RoPE。
    支持两种实现：'naive'（显式缓存 K/V）和 'absorb'（隐式吸收 KV 投影）。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size  # 本 rank 负责的头数
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Q 投影：若 q_lora_rank>0，则使用两阶段 LoRA
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # KV 投影：先降维到 kv_lora_rank + rope_dim
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # 再投影到 (nope + value) 空间
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        # 输出投影
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # 注意力缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            # YaRN 补偿缩放
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 缓存机制
        if attn_impl == "naive":
            # 显式缓存 K/V
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.v_head_dim), persistent=False)
        else:
            # 隐式缓存：只缓存低秩 KV 和 RoPE 部分
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
                                 persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        MLA 前向传播。
        支持增量推理（通过 start_pos 和缓存）。
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Q 投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)  # 对 RoPE 部分应用旋转

        # KV 投影
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # 扩展维度以匹配头数

        if attn_impl == "naive":
            # 显式实现：拼接 nope 和 pe，再投影到完整 K/V
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 更新缓存
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算注意力分数
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 吸收实现：提前加载 wkv_b 权重（可能需反量化）
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight,
                                                                                      self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # Q_nope 与 wkv_b 的 nope 部分相乘
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 更新缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 分别计算 nope 和 pe 的注意力分数
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

        # 应用 mask（仅在 seqlen > 1 时存在）
        if mask is not None:
            scores += mask.unsqueeze(1)
        # Softmax 归一化
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # 加权聚合 V
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 先聚合 kv_cache，再投影到 value 空间
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # 输出投影
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    标准 Dense MLP：SwiGLU 激活。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    MoE 路由门控：计算每个 token 应分配给哪些专家。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 特殊情况：当 dim=7168 时不加偏置（可能是为了兼容某些模型）
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算原始分数
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores  # 保留原始分数用于后续加权

        # 添加偏置（如果存在）
        if self.bias is not None:
            scores = scores + self.bias

        # 分组路由（如果启用）
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)  # 每组最大分
            else:
                # 取 top2 求和作为组分（更鲁棒）
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            # 选择 topk_groups
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # 选择 top-k 专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)  # 归一化
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    单个 MoE 专家：小型 MLP。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    混合专家层（MoE）：支持路由专家 + 共享专家。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(args)
        # 仅初始化本 rank 负责的专家，其余设为 None
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
             for i in range(self.n_routed_experts)])
        # 共享专家（所有 token 都经过）
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)  # 展平 batch 和 seq
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        # 统计每个专家被选中的次数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # 仅处理本 rank 负责的专家
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)  # 聚合各 GPU 的专家输出
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer 块：包含 MLA 注意力和 FFN（Dense 或 MoE）。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        # 前 n_dense_layers 层用 Dense MLP，其余用 MoE
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    完整的 Transformer 模型。
    """

    def __init__(self, args: ModelArgs):
        global world_size, rank
        # 若 PyTorch 分布式已初始化，则更新 world_size 和 rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 设置 Linear 层的默认 dtype 和 scale_fmt
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        Linear.scale_fmt = args.scale_fmt
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        # 输出头：列并行（每个 GPU 负责部分词汇）
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 预计算 RoPE 频率
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        推理模式前向传播。
        支持增量解码（通过 start_pos）。
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            # 生成上三角 mask（因果掩码）
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]  # 取最后一个 token 的表示
        logits = self.head(h)
        if world_size > 1:
            # 聚合各 GPU 的 logits（拼接词汇维度）
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


if __name__ == "__main__":
    # 设置默认 dtype 和 device
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))  # batch=2, seq=128
    model = Transformer(args)
    print(model(x).size())  # 应输出 torch.Size([2, 102400])
