# Simplified GraphCast model using JAX + Haiku
# Based on graphcast.py

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from typing import Tuple, Optional


class GraphCastModel(hk.Module):
    # Simplified GraphCast: Encode-Process-Decode architecture
    # Based on the original GraphCast but with: Fewer variables (2 instead of 227), Smaller mesh (162 nodes instead of 40,962), Fewer layers (6 instead of 16), Single-scale mesh (no multi-mesh)

    def __init__(
        self,
        grid_dim: int = 2,  # geopotential + temperature
        mesh_dim: int = 3,  # xyz coordinates
        latent_dim: int = 128,  # hidden dimension
        num_gnn_layers: int = 6,  # processor depth
        name: str = "graphcast",
    ):
        super().__init__(name=name)
        self.grid_dim = grid_dim
        self.mesh_dim = mesh_dim
        self.latent_dim = latent_dim
        self.num_gnn_layers = num_gnn_layers

    def __call__(
        self,
        grid_input: jnp.ndarray,  # [num_grid, grid_dim]
        mesh_graph: jraph.GraphsTuple,  # Mesh graph structure
        g2m_indices: jnp.ndarray,  # [num_mesh, k] grid-to-mesh mapping
        g2m_weights: jnp.ndarray,  # [num_mesh, k] interpolation weights
        m2g_indices: jnp.ndarray,  # [num_grid, k] mesh-to-grid mapping
        m2g_weights: jnp.ndarray,  # [num_grid, k] interpolation weights
    ) -> jnp.ndarray:
        # Single forward pass
        # Returns:
        #     grid_output: [num_grid, grid_dim] predicted state

        # Encoder: Grid to Mesh
        mesh_features = self._encode(
            grid_input, mesh_graph.nodes, g2m_indices, g2m_weights
        )
        # Update graph with encoded features
        mesh_graph = mesh_graph._replace(nodes=mesh_features)
        # Processor: GNN on mesh
        processed_graph = self._process(mesh_graph)
        # Decoder: Mesh to Grid
        grid_output = self._decode(processed_graph.nodes, m2g_indices, m2g_weights)

        return grid_output

    def _encode(
        self,
        grid_data: jnp.ndarray,  # [num_grid, grid_dim]
        mesh_nodes: jnp.ndarray,  # [num_mesh, mesh_dim] - xyz coords
        g2m_indices: jnp.ndarray,  # [num_mesh, k]
        g2m_weights: jnp.ndarray,  # [num_mesh, k]
    ) -> jnp.ndarray:

        # Encoder: Pool grid data to mesh nodes
        # Based on GraphCast grid2mesh operation

        # Process grid data with MLP
        grid_mlp = hk.nets.MLP(
            output_sizes=[self.latent_dim, self.latent_dim],
            activation=jax.nn.silu,
            name="grid_encoder_mlp",
        )
        grid_processed = grid_mlp(grid_data)

        # Ensure grid_processed is 2D
        grid_processed_flat = grid_processed.reshape(-1, self.latent_dim)

        # Vectorized pooling to mesh
        # Gather neighbors: [num_mesh, k, latent_dim]
        neighbors = grid_processed_flat[g2m_indices]

        # Apply weights: [num_mesh, k, 1] * [num_mesh, k, latent_dim]
        weighted = neighbors * g2m_weights[:, :, jnp.newaxis]

        # Sum over neighbors: [num_mesh, latent_dim]
        mesh_pooled = jnp.sum(weighted, axis=1)

        # Combine with mesh node features (xyz coordinates)
        combined = jnp.concatenate([mesh_pooled, mesh_nodes], axis=-1)

        # Process with MLP
        combine_mlp = hk.nets.MLP(
            output_sizes=[self.latent_dim, self.latent_dim],
            activation=jax.nn.silu,
            name="mesh_encoder_mlp",
        )
        mesh_features = combine_mlp(combined)

        return mesh_features

    def _process(self, mesh_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Processor: Apply GNN layers on mesh
        # Based on GraphCast's GraphNetwork processor
        # Create GNN using Jraph
        def update_edge_fn(edges, senders, receivers, globals_):
            # Edge update function
            # Concatenate sender, receiver, and edge features
            edge_features = jnp.concatenate([senders, receivers, edges], axis=-1)

            # Process with MLP
            edge_mlp = hk.nets.MLP(
                output_sizes=[self.latent_dim], activation=jax.nn.silu, name="edge_mlp"
            )
            return edge_mlp(edge_features)

        def update_node_fn(nodes, sent_attributes, received_attributes, globals_):
            # Node update function
            # Aggregate received messages
            aggregated = received_attributes

            # Concatenate with current node features
            node_features = jnp.concatenate([nodes, aggregated], axis=-1)

            # Update with MLP
            node_mlp = hk.nets.MLP(
                output_sizes=[self.latent_dim, self.latent_dim],
                activation=jax.nn.silu,
                name="node_mlp",
            )
            updated = node_mlp(node_features)

            # Residual connection
            return nodes + updated

        # Apply GNN layers
        graph = mesh_graph
        for layer_idx in range(self.num_gnn_layers):
            with hk.experimental.name_scope(f"gnn_layer_{layer_idx}"):
                # Create graph network for this layer
                gn = jraph.GraphNetwork(
                    update_edge_fn=update_edge_fn,
                    update_node_fn=update_node_fn,
                    aggregate_edges_for_nodes_fn=jraph.segment_sum,
                )

                # Apply layer
                graph = gn(graph)

        return graph

    def _decode(
        self,
        mesh_features: jnp.ndarray,  # [num_mesh, latent_dim]
        m2g_indices: jnp.ndarray,  # [num_grid, k]
        m2g_weights: jnp.ndarray,  # [num_grid, k]
    ) -> jnp.ndarray:

        # Decoder: Unpool mesh features to grid
        # Based on GraphCast mesh2grid operation

        # Vectorized unpooling from mesh to grid
        # Gather neighbors: [num_grid, k, latent_dim]
        neighbors = mesh_features[m2g_indices]

        # Apply weights: [num_grid, k, 1] * [num_grid, k, latent_dim]
        weighted = neighbors * m2g_weights[:, :, jnp.newaxis]

        # Sum over neighbors: [num_grid, latent_dim]
        grid_unpooled = jnp.sum(weighted, axis=1)

        # Decode to grid variables
        decoder_mlp = hk.nets.MLP(
            output_sizes=[self.latent_dim, self.grid_dim],
            activation=jax.nn.silu,
            name="decoder_mlp",
        )
        grid_output = decoder_mlp(grid_unpooled)

        return grid_output


def build_model_forward():
    # Build the forward function (Haiku pattern)
    def forward(
        grid_input, mesh_graph, g2m_indices, g2m_weights, m2g_indices, m2g_weights
    ):
        model = GraphCastModel(grid_dim=2, mesh_dim=3, latent_dim=128, num_gnn_layers=6)
        return model(
            grid_input, mesh_graph, g2m_indices, g2m_weights, m2g_indices, m2g_weights
        )

    return forward


def autoregressive_rollout(
    forward_fn,
    params,
    rng,
    initial_state: jnp.ndarray,
    num_steps: int,
    mesh_graph: jraph.GraphsTuple,
    g2m_indices: jnp.ndarray,
    g2m_weights: jnp.ndarray,
    m2g_indices: jnp.ndarray,
    m2g_weights: jnp.ndarray,
) -> jnp.ndarray:

    # Autoregressive rollout for multiple timesteps\
    # Args:
    #     forward_fn: Haiku transformed forward function (apply method)
    #     params: Model parameters
    #     rng: JAX random key
    #     initial_state: [num_grid, grid_dim]
    #     num_steps: number of steps to predict
    # Returns:
    #     predictions: [num_steps, num_grid, grid_dim]

    predictions = []
    current_state = initial_state

    for step in range(num_steps):
        # Predict next state using EXISTING parameters
        # forward_fn should be the .apply() method
        next_state = forward_fn(
            params,
            rng,
            current_state,
            mesh_graph,
            g2m_indices,
            g2m_weights,
            m2g_indices,
            m2g_weights,
        )

        predictions.append(next_state)

        # Use prediction as input for next step
        current_state = next_state

    return jnp.stack(predictions, axis=0)


if __name__ == "__main__":
    print("GraphCast model definition loaded successfully")
    print("\nThis file defines the model architecture")
    print("Use test_model.py to test the model with actual data")
