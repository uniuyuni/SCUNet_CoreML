from __future__ import annotations

import torch
import torch.nn as nn


class WMSA(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int, type: str, input_resolution: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.head_dim = int(head_dim)
        self.scale = self.head_dim ** -0.5
        self.n_heads = self.input_dim // self.head_dim
        self.window_size = int(window_size)
        self.type = type
        self.input_resolution = int(input_resolution)
        self.h_windows = self.input_resolution // self.window_size
        self.w_windows = self.input_resolution // self.window_size
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1))
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"), dim=-1)
        coords = coords.reshape(-1, 2)
        rel = coords[:, None, :] - coords[None, :, :] + window_size - 1
        relative_index = rel[..., 0] * (2 * window_size - 1) + rel[..., 1]
        self.register_buffer("relative_index", relative_index.reshape(-1).long(), persistent=False)

        if self.type == "SW":
            self.register_buffer(
                "attn_bias",
                self._make_shift_bias(input_resolution, window_size, window_size // 2),
                persistent=False,
            )
        else:
            self.register_buffer("attn_bias", torch.zeros(1, 1, 1, window_size * window_size, window_size * window_size), persistent=False)

    @staticmethod
    def _make_shift_bias(input_resolution: int, window_size: int, shift: int) -> torch.Tensor:
        h_windows = input_resolution // window_size
        w_windows = input_resolution // window_size
        mask = torch.zeros(h_windows, w_windows, window_size, window_size, window_size, window_size)
        s = window_size - shift
        mask[-1, :, :s, :, s:, :] = -100.0
        mask[-1, :, s:, :, :s, :] = -100.0
        mask[:, -1, :, :s, :, s:] = -100.0
        mask[:, -1, :, s:, :, :s] = -100.0
        mask = mask.permute(0, 1, 2, 3, 4, 5).reshape(1, h_windows * w_windows, 1, window_size * window_size, window_size * window_size)
        return mask

    def relative_embedding(self) -> torch.Tensor:
        table = self.relative_position_params.reshape(self.n_heads, -1)
        bias = table[:, self.relative_index]
        return bias.reshape(self.n_heads, self.window_size * self.window_size, self.window_size * self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.window_size
        if self.type == "SW":
            x = torch.roll(x, shifts=(-p // 2, -p // 2), dims=(1, 2))

        x = x.reshape(-1, self.h_windows, p, self.w_windows, p, self.input_dim).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.h_windows * self.w_windows, p * p, self.input_dim)

        qkv = self.embedding_layer(x)
        q, k, v = torch.split(qkv, self.input_dim, dim=-1)
        q = q.reshape(-1, self.h_windows * self.w_windows, p * p, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(-1, self.h_windows * self.w_windows, p * p, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(-1, self.h_windows * self.w_windows, p * p, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        sim = sim + self.relative_embedding().reshape(1, 1, self.n_heads, p * p, p * p)
        sim = sim + self.attn_bias
        probs = torch.softmax(sim, dim=-1)
        out = torch.matmul(probs, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(-1, self.h_windows * self.w_windows, p * p, self.input_dim)
        out = self.linear(out)
        out = out.reshape(-1, self.h_windows, self.w_windows, p, p, self.output_dim).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(-1, self.input_resolution, self.input_resolution, self.output_dim)

        if self.type == "SW":
            out = torch.roll(out, shifts=(p // 2, p // 2), dims=(1, 2))
        return out


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int, type: str, input_resolution: int):
        super().__init__()
        self.type = "W" if input_resolution <= window_size else type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type, input_resolution)
        self.drop_path = nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 4 * input_dim), nn.GELU(), nn.Linear(4 * input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim: int, trans_dim: int, head_dim: int, window_size: int, type: str, input_resolution: int):
        super().__init__()
        self.conv_dim = int(conv_dim)
        self.trans_dim = int(trans_dim)
        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, type, input_resolution)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.conv1_1(x)
        conv_x, trans_x = torch.split(mixed, (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = trans_x.permute(0, 2, 3, 1)
        trans_x = self.trans_block(trans_x)
        trans_x = trans_x.permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class SCUNet(nn.Module):
    def __init__(self, in_nc: int = 3, config: list[int] | tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4), dim: int = 64, input_resolution: int = 448):
        super().__init__()
        self.config = tuple(int(x) for x in config)
        self.dim = int(dim)
        self.head_dim = 32
        self.window_size = 8
        resolution = int(input_resolution)

        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution) for i in range(self.config[0])],
            nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False),
        )
        self.m_down2 = nn.Sequential(
            *[ConvTransBlock(dim, dim, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution // 2) for i in range(self.config[1])],
            nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False),
        )
        self.m_down3 = nn.Sequential(
            *[ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution // 4) for i in range(self.config[2])],
            nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False),
        )
        self.m_body = nn.Sequential(
            *[ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution // 8) for i in range(self.config[3])]
        )
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
            *[ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution // 4) for i in range(self.config[4])],
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
            *[ConvTransBlock(dim, dim, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution // 2) for i in range(self.config[5])],
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
            *[ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, "W" if not i % 2 else "SW", resolution) for i in range(self.config[6])],
        )
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def build_scunet_color_real_psnr(tile: int = 448) -> SCUNet:
    return SCUNet(in_nc=3, config=(4, 4, 4, 4, 4, 4, 4), dim=64, input_resolution=tile)
