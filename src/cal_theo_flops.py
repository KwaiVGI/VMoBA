"""
Calculate the theoratical number of parameters and FLOPs of arbitrary modules.
Ignore batchsize (Assume all batchsize to be 1),
"""

from cycler import V


def format_large_number(num):
    """
    Format a large integer with appropriate suffix (K, M, G, etc.).
    """
    if num < 1_000:
        return f"{num}"  # No suffix for numbers below 1000
    elif num < 1_000_000:
        return f"{num / 1_000:.3f}K"  # Thousands
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.3f}M"  # Millions
    elif num < 1_000_000_000_000:
        return f"{num / 1_000_000_000:.3f}G"  # Billions
    # elif num < 1_000_000_000_000_000:
    else:
        return f"{num / 1_000_000_000_000:.3f}T"  # Trillions
    # else:
    #     return f"{num / 1_000_000_000_000_000:.3f}P"   # Peta FLOPs


class BaseModule():
    def __init__(self):
        self.num_params = 0

class Linear(BaseModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_params = input_dim * output_dim

    def cal_flops(
        self,
        sequence_length: int
    ):
        return 2 * sequence_length * self.input_dim * self.output_dim

class Conv1D(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.num_params = in_channels * out_channels * kernel_size

    def cal_flops(
        self,
        sequence_length: int
    ):
        return 2 * sequence_length * self.in_channels * self.out_channels * self.kernel_size / self.groups

class Matmul(BaseModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.num_params = 0

    def cal_flops(
        self,
        a_row: int,
        a_col: int,
        b_row: int,
        b_col: int,
    ):
        assert a_col == b_row, f"a_row: {a_row}, b_row: {b_row}. Matrix multiplication is not possible."
        return 2 * a_row * b_row * b_col

class SelfAttention(BaseModule):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        output_dim: int,
        use_moba: bool = False,
        moba_chunk_size: int = 2048,
        moba_topk: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.use_moba = use_moba
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk

        self.to_qkv = [Linear(input_dim, inner_dim) for _ in range(3)]
        self.output_proj = Linear(inner_dim, output_dim)

        self.num_params = sum([m.num_params for m in self.to_qkv]) + self.output_proj.num_params

    def cal_flops(
        self,
        sequence_length: int
    ):
        flops = 0
        flops += self.to_qkv[0].cal_flops(sequence_length) * 3
        if self.use_moba:
            flops += Matmul().cal_flops(sequence_length, self.inner_dim, self.inner_dim, sequence_length // self.moba_chunk_size)
            flops += Matmul().cal_flops(sequence_length, self.inner_dim, self.inner_dim, self.moba_chunk_size * self.moba_topk)
            flops += Matmul().cal_flops(sequence_length, self.moba_chunk_size * self.moba_topk, self.moba_chunk_size * self.moba_topk, self.inner_dim)
        else:
            flops += Matmul().cal_flops(sequence_length, self.inner_dim, self.inner_dim, sequence_length)
            flops += Matmul().cal_flops(sequence_length, sequence_length, sequence_length, self.inner_dim)
        flops += self.output_proj.cal_flops(sequence_length)
        return flops
    
class CrossAttention(BaseModule):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim

        self.to_q = Linear(input_dim, inner_dim)
        self.to_kv = [Linear(input_dim, inner_dim) for _ in range(2)]
        self.output_proj = Linear(inner_dim, output_dim)

        self.num_params = self.to_q.num_params + sum([m.num_params for m in self.to_kv]) + self.output_proj.num_params

    def cal_flops(
        self,
        sequence_length: int,
        kv_sequence_length: int
    ):
        flops = 0
        flops += self.to_q.cal_flops(sequence_length)
        flops += self.to_kv[0].cal_flops(kv_sequence_length) * 2
        flops += Matmul().cal_flops(sequence_length, self.inner_dim, self.inner_dim, kv_sequence_length)
        flops += Matmul().cal_flops(sequence_length, kv_sequence_length, kv_sequence_length, self.inner_dim)
        flops += self.output_proj.cal_flops(sequence_length)
        return flops
    
class SelfCrossFFNLayer(BaseModule):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        use_moba: bool = False,
        moba_chunk_size: int = 2048,
        moba_topk: int = 3,
    ):
        super().__init__()
        self.dim = dim

        self.self_attention = SelfAttention(dim, dim, dim, use_moba, moba_chunk_size, moba_topk)
        self.cross_attention = CrossAttention(dim, dim, dim)
        self.ff1 = Linear(dim, ffn_dim)
        self.ff2 = Linear(ffn_dim, dim)

        self.num_params = self.self_attention.num_params + self.cross_attention.num_params + self.ff1.num_params + self.ff2.num_params

    def cal_flops(
        self,
        sequence_length: int,
        kv_sequence_length: int
    ):
        flops = 0
        flops += self.self_attention.cal_flops(sequence_length)
        flops += self.cross_attention.cal_flops(sequence_length, kv_sequence_length)
        flops += self.ff1.cal_flops(sequence_length)
        flops += self.ff2.cal_flops(sequence_length)
        return flops
    


def main(
    sequence_length: int = 320 / (8 * 2) * 512 / (8 * 2) * 84 / 4,
    kv_sequence_length: int = 300
):
    num_layers = 30
    num_attention_heads = 12
    ffn_dim = 8960
    model = SelfCrossFFNLayer(
        dim=num_attention_heads * 128,
        ffn_dim=ffn_dim,
    )
    print(f"WanTransformer information under\nnum_attention_heads={num_attention_heads}\nsequence_length={sequence_length}\nnum_layers={num_layers}:")
    print(f"Model parameters: {format_large_number(model.num_params * num_layers)}")
    print(f"Forward FLOPs: {format_large_number(model.cal_flops(sequence_length, kv_sequence_length) * num_layers)}")

    model = SelfCrossFFNLayer(
        dim=num_attention_heads * 128,
        ffn_dim=ffn_dim,
        use_moba=True,
        moba_chunk_size=sequence_length / 60,
        moba_topk=15
    )
    print(f"MoBA Forward FLOPs: {format_large_number(model.cal_flops(sequence_length, kv_sequence_length) * num_layers)}")

    num_blocks = [7, 40, 60]
    topks = [x * 0.18 for x in num_blocks]
    
    vmoba_1d = SelfCrossFFNLayer(
        dim=num_attention_heads * 128,
        ffn_dim=ffn_dim,
        use_moba=True,
        moba_chunk_size=sequence_length / num_blocks[0],
        moba_topk=topks[0]
    )
    vmoba_2d = SelfCrossFFNLayer(
        dim=num_attention_heads * 128,
        ffn_dim=ffn_dim,
        use_moba=True,
        moba_chunk_size=sequence_length / num_blocks[1],
        moba_topk=topks[1]
    )
    vmoba_3d = SelfCrossFFNLayer(
        dim=num_attention_heads * 128,
        ffn_dim=ffn_dim,
        use_moba=True,
        moba_chunk_size=sequence_length / num_blocks[2],
        moba_topk=topks[2]
    )
    flops = vmoba_1d.cal_flops(sequence_length, kv_sequence_length) * num_layers // 3 + \
            vmoba_2d.cal_flops(sequence_length, kv_sequence_length) * num_layers // 3 + \
            vmoba_3d.cal_flops(sequence_length, kv_sequence_length) * num_layers // 3 
    print(f"VMoBA Forward FLOPs: {format_large_number(flops)}")

    model_svg = SelfCrossFFNLayer(
        dim=num_attention_heads * 128,
        ffn_dim=ffn_dim,
        use_moba=True,
        moba_chunk_size=sequence_length / 64,
        moba_topk=32
    )
    print(f"SVG Forward FLOPs: {format_large_number(model_svg.cal_flops(sequence_length, kv_sequence_length) * num_layers)}")


if __name__ == "__main__":
    """
    python scripts/flops/cal_theo_flops.py
    """
    main()
