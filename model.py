from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class FeatureWiseEmbedding(nn.Module):
    """Embed each variable independently, then fuse to d_model."""

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        d_model: int,
        dropout: float,
        feature_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.feature_dropout = float(feature_dropout)

        if self.num_features == 1:
            self.embed = nn.Conv1d(1, embed_dim, 1)
            embedded_channels = embed_dim
        else:
            self.embed = nn.Conv1d(
                self.num_features,
                self.num_features * embed_dim,
                1,
                groups=self.num_features,
            )
            embedded_channels = self.num_features * embed_dim

        self.fuse = nn.Sequential(
            nn.Conv1d(embedded_channels, d_model, 1),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # [B,N,C] -> [B,C,N]
        x = x.transpose(1, 2).contiguous()

        if (
            self.training
            and self.feature_dropout > 0
            and self.num_features > 1
        ):
            keep = 1.0 - self.feature_dropout
            mask = torch.empty(
                x.size(0), x.size(1), 1,
                device=x.device, dtype=x.dtype,
            ).bernoulli_(keep)
            x = x * mask / keep

        return self.fuse(self.embed(x))


class ECALayer(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            1, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        weight = x.mean(dim=-1, keepdim=True).transpose(1, 2)
        weight = torch.sigmoid(self.conv(weight)).transpose(1, 2)
        return x * weight


class ConvCAnextLite(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.2,
        drop_path: float = 0.1,
    ):
        super().__init__()
        hidden = d_model * expansion

        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(1, d_model)
        self.pw1 = nn.Conv1d(d_model, hidden, 1)
        self.pw2 = nn.Conv1d(hidden, d_model, 1)
        self.norm2 = nn.GroupNorm(1, d_model)
        self.eca = ECALayer()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.scale = nn.Parameter(1e-3 * torch.ones(1, d_model, 1))

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.dw(x)))
        x = self.pw2(self.dropout(F.gelu(self.pw1(x))))
        x = self.eca(self.norm2(x))
        return residual + self.drop_path(self.scale * x)


class GraphDown(nn.Module):
    def forward(self, h, assignment):
        batch, _, dim = h.shape
        num_coarse = int(assignment.max().item()) + 1

        out = torch.zeros(
            batch, num_coarse, dim,
            device=h.device, dtype=h.dtype,
        )
        out.index_add_(1, assignment, h.to(out.dtype))

        counts = torch.bincount(
            assignment,
            minlength=num_coarse,
        ).to(device=h.device, dtype=h.dtype)

        return out / counts.view(1, -1, 1).clamp_min(1)


class NormalizedGraphMPNN(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        edge_dropout: float,
        edge_chunk_size: int,
        use_destination: bool = False,
    ):
        super().__init__()
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_dropout = float(edge_dropout)
        self.use_destination = bool(use_destination)

        self.src = nn.Linear(d_model, d_model, bias=False)
        self.dst = (
            nn.Linear(d_model, d_model, bias=False)
            if self.use_destination else None
        )
        self.edge = nn.Linear(d_model, d_model)
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(1e-3 * torch.ones(1, 1, d_model))

    def forward(self, h, edge_index, edge_hidden):
        src, dst = edge_index
        agg = torch.zeros_like(h)
        degree = torch.bincount(
            dst, minlength=h.size(1)
        ).to(device=h.device, dtype=h.dtype).clamp_min(1)

        edge_hidden = self.edge(edge_hidden)

        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            s = src[start:end]
            d = dst[start:end]

            msg = self.src(h.index_select(1, s))
            if self.dst is not None:
                msg = msg + self.dst(h.index_select(1, d))
            msg = F.gelu(msg + edge_hidden[start:end].unsqueeze(0))

            if self.training and self.edge_dropout > 0:
                keep = 1.0 - self.edge_dropout
                mask = torch.empty(
                    1, msg.size(1), 1,
                    device=msg.device, dtype=msg.dtype,
                ).bernoulli_(keep)
                msg = msg * mask / keep

            agg.index_add_(1, d, msg.to(agg.dtype))

        agg = agg / degree.view(1, -1, 1)
        update = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + self.scale * update)


class CoarseGraphDiffusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        edge_dropout: float,
        edge_chunk_size: int,
        initial_gate: float = -2.0,
    ):
        super().__init__()
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_dropout = float(edge_dropout)
        self.edge_gate = nn.Linear(d_model, 1)
        self.mix = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, h, edge_index, edge_hidden):
        src, dst = edge_index
        weights = torch.sigmoid(self.edge_gate(edge_hidden).squeeze(-1))
        agg = torch.zeros_like(h)

        degree = torch.bincount(
            dst, minlength=h.size(1)
        ).to(device=h.device, dtype=h.dtype).clamp_min(1)

        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            s = src[start:end]
            d = dst[start:end]

            msg = h.index_select(1, s)
            msg = msg * weights[start:end].view(1, -1, 1)

            if self.training and self.edge_dropout > 0:
                keep = 1.0 - self.edge_dropout
                mask = torch.empty(
                    1, msg.size(1), 1,
                    device=msg.device, dtype=msg.dtype,
                ).bernoulli_(keep)
                msg = msg * mask / keep

            agg.index_add_(1, d, msg.to(agg.dtype))

        agg = self.mix(agg / degree.view(1, -1, 1))
        return self.norm(h + torch.sigmoid(self.gate) * agg)


class SimpleFNO2d(nn.Module):
    def __init__(self, d_model: int, modes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.modes = int(modes)
        self.weight = nn.Parameter(
            0.02 * torch.randn(d_model, modes, modes, 2)
        )
        self.post = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model, d_model, 1),
        )

    def forward(self, x):
        original_dtype = x.dtype
        _, _, height, width = x.shape

        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            x32 = torch.nan_to_num(
                x.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(-1e4, 1e4)

            spectrum = torch.fft.rfft2(
                x32,
                norm="ortho",
            )
            out = torch.zeros_like(spectrum)
            weight = torch.view_as_complex(
                self.weight.float()
            )

            mx = min(self.modes, height)
            my = min(self.modes, spectrum.size(-1))
            out[:, :, :mx, :my] = (
                spectrum[:, :, :mx, :my]
                * weight[:, :mx, :my]
            )

            output = torch.fft.irfft2(
                out,
                s=(height, width),
                norm="ortho",
            )
            output = self.post.float()(output)
            output = torch.nan_to_num(
                output,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(-1e4, 1e4)

        return output.to(original_dtype)



class GINOLite(nn.Module):
    def __init__(self, d_model: int, grid_res: int, dropout: float):
        super().__init__()
        self.grid_res = int(grid_res)
        self.fno = SimpleFNO2d(
            d_model,
            modes=min(8, self.grid_res),
            dropout=dropout,
        )

    def forward(self, h, pos):
        batch, _, dim = h.shape
        pos = pos.to(device=h.device, dtype=h.dtype)

        pmin = pos.min(dim=0, keepdim=True).values
        pmax = pos.max(dim=0, keepdim=True).values
        norm_pos = (pos - pmin) / (pmax - pmin).clamp_min(1e-6)

        coords = (norm_pos * (self.grid_res - 1)).long()
        index = coords[:, 0] * self.grid_res + coords[:, 1]
        cells = self.grid_res * self.grid_res

        grid = torch.zeros(
            batch, dim, cells,
            device=h.device, dtype=h.dtype,
        )
        grid.index_add_(
            2,
            index,
            h.transpose(1, 2).to(grid.dtype),
        )

        counts = torch.bincount(
            index, minlength=cells
        ).to(device=h.device, dtype=h.dtype)
        grid = grid / counts.view(1, 1, -1).clamp_min(1)

        grid = grid.view(
            batch, dim, self.grid_res, self.grid_res
        )
        output = self.fno(grid).flatten(2)
        return output[:, :, index].transpose(1, 2).contiguous()


class LinearAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        original_dtype = tokens.dtype

        with torch.autocast(
            device_type=tokens.device.type,
            enabled=False,
        ):
            tokens32 = torch.nan_to_num(
                tokens.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(-1e4, 1e4)

            q, k, v = self.qkv.float()(
                tokens32
            ).chunk(3, dim=-1)

            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0

            q = q / q.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
            k = k / k.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)

            kv = torch.einsum(
                "bmd,bme->bde",
                k,
                v,
            )
            numerator = torch.einsum(
                "bmd,bde->bme",
                q,
                kv,
            )
            denominator = torch.einsum(
                "bmd,bd->bm",
                q,
                k.sum(dim=1),
            ).unsqueeze(-1)

            output = numerator / denominator.clamp_min(
                1e-4
            )
            output = self.out.float()(
                self.dropout(output)
            )
            output = torch.nan_to_num(
                output,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(-1e4, 1e4)

        return output.to(original_dtype)



class PartitionTokenAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        initial_gate: float = -2.0,
    ):
        super().__init__()
        self.attn = LinearAttention(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, h, assignment):
        batch, _, dim = h.shape
        num_tokens = int(assignment.max().item()) + 1

        tokens = torch.zeros(
            batch, num_tokens, dim,
            device=h.device, dtype=h.dtype,
        )
        tokens.index_add_(
            1,
            assignment,
            h.to(tokens.dtype),
        )

        counts = torch.bincount(
            assignment, minlength=num_tokens
        ).to(device=h.device, dtype=h.dtype)

        tokens = tokens / counts.view(1, -1, 1).clamp_min(1)
        tokens = self.attn(tokens)
        token_up = tokens.index_select(1, assignment)

        return self.norm(
            h + torch.sigmoid(self.gate) * token_up
        )


class CausalTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.dw = nn.Conv2d(
            d_model, d_model,
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1),
            groups=d_model,
            bias=False,
        )
        self.gate = nn.Conv2d(d_model, 2 * d_model, 1)
        self.out = nn.Conv2d(d_model, d_model, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.GroupNorm(1, d_model)
        self.drop_path = DropPath(drop_path)
        self.scale = nn.Parameter(
            1e-3 * torch.ones(1, d_model, 1, 1)
        )

    def forward(self, x):
        residual = x
        x = F.pad(x, (0, 0, self.pad, 0))
        x = self.dw(x)
        value, gate = self.gate(x).chunk(2, dim=1)
        x = torch.tanh(value) * torch.sigmoid(gate)
        x = self.out(self.dropout(x))
        return self.norm(
            residual + self.drop_path(self.scale * x)
        )


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        rates = torch.linspace(0, drop_path, layers).tolist()
        self.blocks = nn.ModuleList([
            CausalTemporalBlock(
                d_model,
                kernel_size,
                2 ** i,
                dropout,
                rates[i],
            )
            for i in range(layers)
        ])
        self.pool_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, sequence):
        # [B,T,N,D] -> [B,D,T,N]
        x = sequence.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)

        last = x[:, :, -1]
        mean = x.mean(dim=2)
        gate = torch.sigmoid(self.pool_gate)
        return (
            gate * last + (1 - gate) * mean
        ).transpose(1, 2).contiguous()


class DirectHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        horizon: int,
        output_dim: int,
        dropout: float,
        ratio: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        hidden = max(output_dim, int(d_model * ratio))

        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, horizon * output_dim, 1),
        )

    def forward(self, h):
        batch, nodes, _ = h.shape
        output = self.net(h.transpose(1, 2))
        output = output.view(
            batch, self.horizon, self.output_dim, nodes
        )
        return output.permute(0, 1, 3, 2).contiguous()


class HOPNet(nn.Module):
    """Hybrid direct multi-horizon CHONet."""

    def __init__(self, cfg):
        super().__init__()

        self.state_dim = int(cfg.get("state_dim", cfg.get("input_dim", 4)))
        self.output_dim = int(cfg.get("output_dim", self.state_dim))
        self.static_dim = int(cfg.get("static_dim", 2))

        self.horizon = int(cfg.get("horizon", 12))
        self.d_model = int(cfg.get("d_model", 64))
        self.edge_dim = int(cfg.get("edge_dim", 3))
        self.num_nodes = cfg.get("num_nodes")

        dropout = float(cfg.get("dropout", 0.2))
        drop_path = float(cfg.get("drop_path", 0.1))
        feature_dropout = float(cfg.get("feature_dropout", 0.05))
        edge_dropout = float(cfg.get("edge_dropout", 0.1))

        self.use_persistence = bool(cfg.get("use_persistence", True))
        self.use_gino = bool(cfg.get("use_gino", True))
        self.use_partition_attention = bool(
            cfg.get("use_partition_attention", True)
        )
        self.coarse_mode = str(cfg.get("coarse_mode", "diffusion")).lower()
        self.input_noise_std = float(cfg.get("input_noise_std", 0.005))
        self.spatial_chunk_size = int(cfg.get("spatial_chunk_size", 1))
        self.use_checkpoint = bool(
            cfg.get("use_gradient_checkpointing", True)
        )
        self.use_temporal = bool(cfg.get("use_temporal", True))
        self.temporal_ablation_mode = str(cfg.get("temporal_ablation_mode","last",)).lower()        

        def embed_dim(key, channels):
            requested = int(cfg.get(key, 4))
            return min(requested, max(4, self.d_model // max(1, channels)))

        self.state_embed = FeatureWiseEmbedding(
            self.state_dim,
            embed_dim("state_embed_dim", self.state_dim),
            self.d_model,
            dropout,
            feature_dropout,
        )
        self.static_embed = FeatureWiseEmbedding(
            self.static_dim,
            embed_dim("static_embed_dim", self.static_dim),
            self.d_model,
            dropout,
        )
        self.pos_embed = FeatureWiseEmbedding(
            2,
            embed_dim("position_embed_dim", 2),
            self.d_model,
            dropout,
        )

        self.input_fusion = nn.Sequential(
            nn.Conv1d(3 * self.d_model, self.d_model, 1),
            nn.GroupNorm(1, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.edge_embed = FeatureWiseEmbedding(
            self.edge_dim,
            embed_dim("edge_embed_dim", self.edge_dim),
            self.d_model,
            dropout,
        )

        self.local_encoder = ConvCAnextLite(
            self.d_model,
            expansion=int(cfg.get("expansion_ratio", 2)),
            kernel_size=int(cfg.get("spatial_kernel_size", 5)),
            dropout=dropout,
            drop_path=drop_path,
        )

        self.fine_mpnn = NormalizedGraphMPNN(
            self.d_model,
            dropout,
            edge_dropout,
            int(cfg.get("fine_edge_chunk_size", 2048)),
            use_destination=bool(cfg.get("fine_use_destination", False)),
        )
        self.down = GraphDown()

        if self.coarse_mode == "diffusion":
            self.coarse_block = CoarseGraphDiffusion(
                self.d_model,
                dropout,
                edge_dropout,
                int(cfg.get("coarse_edge_chunk_size", 1024)),
                float(cfg.get("coarse_gate_init", -2.0)),
            )
        elif self.coarse_mode == "mpnn":
            self.coarse_block = NormalizedGraphMPNN(
                self.d_model,
                dropout,
                edge_dropout,
                int(cfg.get("coarse_edge_chunk_size", 1024)),
                use_destination=True,
            )
        elif self.coarse_mode == "none":
            self.coarse_block = None
        else:
            raise ValueError("coarse_mode must be diffusion, mpnn, or none.")

        self.coarse_up_gate = nn.Parameter(
            torch.tensor(float(cfg.get("coarse_up_gate_init", -2.0)))
        )

        if self.use_gino:
            grid_res = int(cfg.get(
                "grid_res",
                max(16, int((self.num_nodes or 4096) ** 0.5)),
            ))
            self.gino = GINOLite(self.d_model, grid_res, dropout)
            self.gino_gate = nn.Parameter(
                torch.tensor(float(cfg.get("gino_gate_init", -2.0)))
            )
        else:
            self.gino = None
            self.gino_gate = None

        self.partition_attention = (
            PartitionTokenAttention(
                self.d_model,
                dropout,
                float(cfg.get("partition_gate_init", -2.0)),
            )
            if self.use_partition_attention
            else None
        )

        if self.use_temporal:
            self.temporal = TemporalEncoder(
                self.d_model,
                int(cfg.get("temporal_layers", 2)),
                int(cfg.get("temporal_kernel_size", 3)),
                dropout,
                drop_path,
            )
        else:
            self.temporal = None        

        head_dropout = float(cfg.get("head_dropout", 0.25))
        head_ratio = float(cfg.get("head_bottleneck_ratio", 0.5))

        self.state_head = DirectHead(
            self.d_model, self.horizon,
            self.output_dim, head_dropout, head_ratio,
        )
        if self.use_persistence and self.output_dim > self.state_dim:
            raise ValueError(
                "Persistence requires output_dim <= state_dim."
            )

    def _encode_edges(self, edge_attr):
        edge_attr = torch.nan_to_num(
            edge_attr,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        encoded = self.edge_embed(edge_attr.unsqueeze(0))
        return encoded.squeeze(0).transpose(0, 1).contiguous()

    def _embed_inputs(self, state, graph):
        batch = state.size(0)

        state = torch.nan_to_num(
            state,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        static = torch.nan_to_num(
            graph.static_node.to(
                device=state.device,
                dtype=state.dtype,
            ),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).unsqueeze(0).expand(batch, -1, -1)

        pos = torch.nan_to_num(
            graph.pos.to(
                device=state.device,
                dtype=state.dtype,
            ),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).unsqueeze(0).expand(batch, -1, -1)

        hidden = torch.cat(
            [
                self.state_embed(state),
                self.static_embed(static),
                self.pos_embed(pos),
            ],
            dim=1,
        )

        return self.input_fusion(hidden)

    def _spatial_chunk(
        self,
        state,
        fine_edge_hidden,
        coarse_edge_hidden,
        graph,
    ):
        h = self._embed_inputs(state, graph)
        h = self.local_encoder(h).transpose(1, 2).contiguous()

        fine = self.fine_mpnn(
            h, graph.edge_index, fine_edge_hidden
        )

        if self.coarse_block is not None:
            coarse = self.down(fine, graph.level_assignment)
            coarse = self.coarse_block(
                coarse,
                graph.coarse_edge_index,
                coarse_edge_hidden,
            )
            coarse_up = coarse.index_select(
                1, graph.level_assignment
            )
            h = fine + torch.sigmoid(
                self.coarse_up_gate
            ) * coarse_up
        else:
            h = fine

        if self.gino is not None:
            h = h + torch.sigmoid(
                self.gino_gate
            ) * self.gino(h, graph.pos)

        if self.partition_attention is not None:
            h = self.partition_attention(
                h, graph.level_assignment
            )

        return h

    def _spatial_sequence(
        self,
        history_state,
        graph,
    ):
        batch, steps, nodes, _ = history_state.shape

        if self.training and self.input_noise_std > 0:
            history_state = history_state + (
                torch.randn_like(history_state)
                * self.input_noise_std
            )

        state = history_state.reshape(
            batch * steps,
            nodes,
            self.state_dim,
        )

        fine_edge_hidden = self._encode_edges(
            graph.edge_attr
        )
        coarse_edge_hidden = self._encode_edges(
            graph.coarse_edge_attr
        )

        outputs = []

        for start in range(
            0,
            state.size(0),
            self.spatial_chunk_size,
        ):
            end = min(
                start + self.spatial_chunk_size,
                state.size(0),
            )

            def spatial_fn(s, ef, ec):
                return self._spatial_chunk(
                    s,
                    ef,
                    ec,
                    graph,
                )

            if (
                self.training
                and self.use_checkpoint
                and torch.is_grad_enabled()
            ):
                encoded = checkpoint(
                    spatial_fn,
                    state[start:end],
                    fine_edge_hidden,
                    coarse_edge_hidden,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                encoded = spatial_fn(
                    state[start:end],
                    fine_edge_hidden,
                    coarse_edge_hidden,
                )

            outputs.append(encoded)

        return torch.cat(outputs, dim=0).view(
            batch,
            steps,
            nodes,
            self.d_model,
        )

    def forward(
        self,
        history_state,
        graph,
    ):
        """
        Parameters
        ----------
        history_state:
            [B,T,N,state_dim]
        graph:
            Fine/coarse CoastalGraph object.

        Returns
        -------
        {"state": [B,H,N,output_dim]}
        """

        sequence = self._spatial_sequence(
            history_state,
            graph,
        )
        if not torch.isfinite(sequence).all():
            raise FloatingPointError(
                "Spatial HOPNet sequence contains NaN or Inf."
            )

        if self.use_temporal:
            hidden = self.temporal(sequence)

        elif self.temporal_ablation_mode == "last":
            hidden = sequence[:, -1]

        elif self.temporal_ablation_mode == "mean":
            hidden = sequence.mean(dim=1)

        elif self.temporal_ablation_mode == "max":
            hidden = sequence.max(dim=1).values

        else:
            raise ValueError(
                "temporal_ablation_mode must be "
                "'last', 'mean', or 'max'."
            )

        if not torch.isfinite(hidden).all():
            raise FloatingPointError(
                "Temporal HOPNet hidden state contains NaN or Inf."
            )

        state_delta = self.state_head(hidden)

        if self.use_persistence:
            baseline = history_state[
                :, -1, :, :self.output_dim
            ].unsqueeze(1)
            state = baseline + state_delta
        else:
            state = state_delta

        if not torch.isfinite(state).all():
            raise FloatingPointError(
                "HOPNet state forecast contains NaN or Inf."
            )

        return {"state": state}

    @torch.no_grad()
    def rollout(
        self,
        history_state,
        graph,
        horizon: Optional[int] = None,
    ):
        """
        True autoregressive rollout.

        The direct model predicts H steps, but rollout feeds only the first
        predicted step back into the history window at each iteration.
        """
        steps = int(
            self.horizon if horizon is None else horizon
        )

        if steps < 1:
            raise ValueError("horizon must be at least 1.")

        current_state = history_state.clone()
        state_outputs = []

        for _ in range(steps):
            prediction = self.forward(
                history_state=current_state,
                graph=graph,
            )

            next_state = prediction["state"][:, 0]
            state_outputs.append(next_state)

            # Preserve non-output channels when input_dim > output_dim.
            next_input = current_state[:, -1].clone()
            next_input[..., :self.output_dim] = next_state

            current_state = torch.cat(
                [
                    current_state[:, 1:],
                    next_input.unsqueeze(1),
                ],
                dim=1,
            )

        return {
            "state": torch.stack(
                state_outputs,
                dim=1,
            )
        }
