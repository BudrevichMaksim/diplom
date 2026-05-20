import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class GATBlock(nn.Module):
    """
    Graph Attention Network residual block using GATv2 layers.

    Integrates multi-head graph attention mechanisms with structural layer normalization,
    GELU activations, and a linear shortcut to safely bridge dimension adjustments.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        """
        Initializes the GATv2 structural block.

        Args:
            in_channels (int): Input feature size per node.
            out_channels (int): Total target output channel representation space.
            heads (int): Number of multi-head attention components.
        """
        super().__init__()
        self.gat = GATv2Conv(
            in_channels, out_channels // heads, heads=heads, dropout=0.1
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

        self.shortcut = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Processes graph nodes through multi-head attention matching.

        Args:
            x (torch.Tensor): Node features matrix of shape [Total_Nodes, In_Channels].
            edge_index (torch.Tensor): Graph topology coordinate pairs of shape [2, Total_Edges].

        Returns:
            torch.Tensor: Attentive node representations of shape [Total_Nodes, Out_Channels].
        """
        residual = self.shortcut(x)
        x = self.gat(x, edge_index)
        x = self.norm(x + residual)
        return self.act(x)


class GATDecoder(nn.Module):
    """
    Graph-based decoder architecture using localized temporal window relationships.

    Transforms sequential frame maps into sparse temporal node graphs, running multi-layered
    graph attention exclusively across localized neighborhood windows to prevent memory inflation,
    followed by global statistical readout pooling.
    """

    def __init__(self, input_size: int, hidden_size: int = 256):
        """
        Initializes the graph attention network decoder.

        Args:
            input_size (int): Flattened channel-frequency footprint slice size (C * H).
            hidden_size (int): Target latent dimension space for node tracking blocks.
        """
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                GATBlock(hidden_size, hidden_size, heads=4),
                GATBlock(hidden_size, hidden_size, heads=4),
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, window_radius: int = 5) -> torch.Tensor:
        """
        Decodes sequential feature configurations using memory-optimized sliding windows.

        Args:
            x (torch.Tensor): Feature maps from encoder of shape [Batch, Channels, Height, Width].
            window_radius (int): Temporal radius defining local node connections.

        Returns:
            torch.Tensor: Unnormalized output logit prediction mapping tensor of shape [Batch].
        """
        b, c, h, w = x.shape

        # 1. Prepare sequential node features
        x = x.permute(0, 3, 1, 2).contiguous()  # [Batch, Width, Channels, Height]
        x = x.view(b, w, c * h)                  # [Batch, Width, Input_Size]
        x = self.embedding(x)
        device = x.device

        x_flat = x.view(-1, x.size(-1))          # [Batch * Width, Hidden_Size]

        # 2. Vectorized Generation of a Sparse Sliding Window Graph (O(W) complexity instead of O(W^2))
        idx = torch.arange(w, device=device)
        idx_i = idx.repeat_interleave(2 * window_radius + 1)
        offsets = torch.arange(
            -window_radius, window_radius + 1, device=device
        ).repeat(w)
        idx_j = idx_i + offsets

        # Filter out boundary violations
        valid_mask = (idx_j >= 0) & (idx_j < w)
        base_edge_index = torch.stack(
            [idx_i[valid_mask], idx_j[valid_mask]], dim=0
        )  # [2, Valid_Edges_Per_Sample]

        # 3. Replicate across batch using broadcasting
        shifts = torch.arange(b, device=device) * w
        edge_index_all = base_edge_index.unsqueeze(1) + shifts.view(1, -1, 1)
        edge_index_all = edge_index_all.view(2, -1)  # Clear mapping: [2, Total_Sparse_Edges]

        # 4. Message passing through GAT layers
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index_all)

        # 5. Global statistical pooling
        batch_vec = torch.arange(b, device=device).repeat_interleave(w)
        out_mean = global_mean_pool(x_flat, batch_vec)
        out_max = global_max_pool(x_flat, batch_vec)
        combined = torch.cat([out_mean, out_max], dim=1)

        return self.classifier(combined).squeeze(1)
