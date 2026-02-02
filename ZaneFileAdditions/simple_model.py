import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import numpy as np


class GraphCastEncoder(nn.Module):
    # Encoder: Grid to Mesh
    # Pools grid data to mesh nodes using learned weights

    def __init__(self, grid_dim, mesh_dim, latent_dim):
        #     grid_dim: number of input variables: 2 for geopotential, temperature
        #     mesh_dim: node feature dimension: 3 for xyz coordinates
        #     latent_dim: 128

        super().__init__()

        # MLP to process grid data before pooling
        self.grid_mlp = nn.Sequential(
            nn.Linear(grid_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # MLP to combine pooled grid data with mesh node features
        self.combine_mlp = nn.Sequential(
            nn.Linear(latent_dim + mesh_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        print(
            f"Encoder: grid_dim={grid_dim}, mesh_dim={mesh_dim}, latent_dim={latent_dim}"
        )

    def forward(self, grid_data, mesh_features, g2m_indices, g2m_weights):
        #     grid_data: [batch, num_grid, grid_dim] - weather data on grid
        #     mesh_features: [num_mesh, mesh_dim] - mesh node xyz coordinates
        #     g2m_indices: [num_mesh, k] - nearest grid points for each mesh node
        #     g2m_weights: [num_mesh, k] - interpolation weights
        # Returns mesh_latent: [batch, num_mesh, latent_dim] - latent features on mesh

        batch_size = grid_data.shape[0]
        num_mesh = g2m_indices.shape[0]

        # Process grid data
        grid_processed = self.grid_mlp(grid_data)

        # Pool to mesh nodes
        mesh_pooled = torch.zeros(
            batch_size, num_mesh, grid_processed.shape[-1], device=grid_data.device
        )

        for i in range(num_mesh):
            neighbors = g2m_indices[i]
            weights = g2m_weights[i]

            # Weighted average of neighbor features
            neighbor_features = grid_processed[
                :, neighbors, :
            ]  # [batch, k, latent_dim]
            weights_expanded = weights.view(1, -1, 1)  # [1, k, 1]
            mesh_pooled[:, i, :] = (neighbor_features * weights_expanded).sum(dim=1)

        # Combine with mesh node features
        mesh_features_expanded = mesh_features.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([mesh_pooled, mesh_features_expanded], dim=-1)

        mesh_latent = self.combine_mlp(combined)

        return mesh_latent


class GNNLayer(MessagePassing):
    # Single GNN layer for message passing on mesh

    def __init__(self, latent_dim, edge_dim):
        super().__init__(aggr="add")  # Use 'add' aggregation

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # Returns x_updated: [num_nodes, latent_dim] - updated node features

        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update nodes
        x_updated = self.node_mlp(torch.cat([x, out], dim=-1))

        # Residual connection
        return x + x_updated

    def message(self, x_i, x_j, edge_attr):
        # Create messages for each edge
        # x_i: [num_edges, latent_dim] - target node features
        # x_j: [num_edges, latent_dim] - source node features
        # edge_attr: [num_edges, edge_dim] - edge features

        # Concatenate source, target, and edge features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Process with edge MLP
        message = self.edge_mlp(edge_input)

        return message


class GraphCastProcessor(nn.Module):
    # Processor: GNN operating on mesh
    def __init__(self, latent_dim, edge_dim, num_layers=6):
        super().__init__()

        self.num_layers = num_layers

        # Stack of GNN layers
        self.layers = nn.ModuleList(
            [GNNLayer(latent_dim, edge_dim) for _ in range(num_layers)]
        )

        print(f"Processor: {num_layers} GNN layers, latent_dim={latent_dim}")

    def forward(self, mesh_latent, edge_index, edge_attr):
        batch_size = mesh_latent.shape[0]

        # Process each sample in batch
        outputs = []
        for b in range(batch_size):
            x = mesh_latent[b]  # [num_mesh, latent_dim]

            # Apply GNN layers
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)

            outputs.append(x)

        mesh_processed = torch.stack(outputs, dim=0)

        return mesh_processed


class GraphCastDecoder(nn.Module):
    # Decoder: Mesh to Grid
    # Unpools mesh features back to grid
    def __init__(self, latent_dim, grid_dim):
        super().__init__()

        # MLP to decode from latent to grid variables
        self.decode_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, grid_dim),
        )

        print(f"Decoder: latent_dim={latent_dim}, grid_dim={grid_dim}")

    def forward(self, mesh_latent, m2g_indices, m2g_weights):
        batch_size = mesh_latent.shape[0]
        num_grid = m2g_indices.shape[0]

        # Unpool from mesh to grid
        grid_latent = torch.zeros(
            batch_size, num_grid, mesh_latent.shape[-1], device=mesh_latent.device
        )

        for i in range(num_grid):
            neighbors = m2g_indices[i]  # [k]
            weights = m2g_weights[i]  # [k]

            # Weighted average of neighbor features
            neighbor_features = mesh_latent[:, neighbors, :]  # [batch, k, latent_dim]
            weights_expanded = weights.view(1, -1, 1)  # [1, k, 1]
            grid_latent[:, i, :] = (neighbor_features * weights_expanded).sum(dim=1)

        # Decode to grid variables
        grid_output = self.decode_mlp(grid_latent)

        return grid_output


class GraphCast(nn.Module):
    # Complete GraphCast model
    def __init__(
        self, grid_dim=2, mesh_dim=3, latent_dim=128, edge_dim=1, num_gnn_layers=6
    ):
        super().__init__()

        print("INITIALIZING GRAPHCAST MODEL")

        self.encoder = GraphCastEncoder(grid_dim, mesh_dim, latent_dim)
        self.processor = GraphCastProcessor(latent_dim, edge_dim, num_gnn_layers)
        self.decoder = GraphCastDecoder(latent_dim, grid_dim)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nModel parameters:")
        print(f"Total: {total_params:,}")
        print(f"Trainable: {trainable_params:,}")

    def forward(
        self,
        grid_input,
        mesh_features,
        edge_index,
        edge_attr,
        g2m_indices,
        g2m_weights,
        m2g_indices,
        m2g_weights,
    ):
        # Single forward pass
        # grid_input: [batch, num_grid, grid_dim]
        # mesh_features: [num_mesh, mesh_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        # g2m_indices, g2m_weights: grid→mesh mapping
        # m2g_indices, m2g_weights: mesh→grid mapping
        # Returns grid_output: [batch, num_grid, grid_dim]

        # Encode: Grid to Mesh
        mesh_latent = self.encoder(grid_input, mesh_features, g2m_indices, g2m_weights)

        # Process: GNN on mesh
        mesh_processed = self.processor(mesh_latent, edge_index, edge_attr)

        # Decode: Mesh to Grid
        grid_output = self.decoder(mesh_processed, m2g_indices, m2g_weights)

        return grid_output

    def autoregressive_rollout(
        self,
        initial_state,
        num_steps,
        mesh_features,
        edge_index,
        edge_attr,
        g2m_indices,
        g2m_weights,
        m2g_indices,
        m2g_weights,
    ):
        # Autoregressive prediction for multiple timesteps
        # Returns predictions: [batch, num_steps, num_grid, grid_dim]
        predictions = []
        current_state = initial_state

        for step in range(num_steps):
            # Predict next state
            next_state = self.forward(
                current_state,
                mesh_features,
                edge_index,
                edge_attr,
                g2m_indices,
                g2m_weights,
                m2g_indices,
                m2g_weights,
            )

            predictions.append(next_state)

            # Use prediction as input for next step
            current_state = next_state

        # Stack predictions
        predictions = torch.stack(
            predictions, dim=1
        )  # [batch, num_steps, num_grid, grid_dim]

        return predictions


if __name__ == "__main__":
    # Test model initialization
    model = GraphCast(
        grid_dim=2,  # geopotential + temperature
        mesh_dim=3,  # xyz coordinates
        latent_dim=128,  # hidden dimension
        edge_dim=1,  # edge distance
        num_gnn_layers=6,  # number of GNN layers
    )

    print("\nModel created successfully")
