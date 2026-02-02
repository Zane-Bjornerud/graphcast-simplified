# Test the simplified GraphCast model
#
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import numpy as np

from simple_graphcast import build_model_forward, autoregressive_rollout


def create_dummy_mesh_graph(num_nodes=162, num_edges=480):
    # Node features (xyz coordinates)
    nodes = jnp.array(np.random.randn(num_nodes, 3).astype(np.float32))

    # Edge features (distances)
    edges = jnp.array(np.random.rand(num_edges, 1).astype(np.float32))

    # Edge connectivity (random for testing)
    senders = jnp.array(np.random.randint(0, num_nodes, num_edges))
    receivers = jnp.array(np.random.randint(0, num_nodes, num_edges))

    # Create graph
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([num_edges]),
        globals=None,
    )

    return graph


def test_model():
    print("TESTING GRAPHCAST MODEL")

    # Create dummy data
    num_grid = 2048  # 32 * 64
    num_mesh = 162
    k_g2m = 4
    k_m2g = 3

    grid_input = jnp.array(np.random.randn(num_grid, 2).astype(np.float32))
    mesh_graph = create_dummy_mesh_graph(num_mesh, 480)

    g2m_indices = jnp.array(np.random.randint(0, num_grid, (num_mesh, k_g2m)))
    g2m_weights = jnp.array(np.random.rand(num_mesh, k_g2m).astype(np.float32))
    g2m_weights = g2m_weights / g2m_weights.sum(axis=1, keepdims=True)

    m2g_indices = jnp.array(np.random.randint(0, num_mesh, (num_grid, k_m2g)))
    m2g_weights = jnp.array(np.random.rand(num_grid, k_m2g).astype(np.float32))
    m2g_weights = m2g_weights / m2g_weights.sum(axis=1, keepdims=True)

    print(f"\nInput shapes:")
    print(f"  Grid input: {grid_input.shape}")
    print(f"  Mesh nodes: {mesh_graph.nodes.shape}")
    print(f"  Mesh edges: {mesh_graph.edges.shape}")

    # Build model
    forward_fn = build_model_forward()
    forward_fn = hk.transform(forward_fn)

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    params = forward_fn.init(
        rng, grid_input, mesh_graph, g2m_indices, g2m_weights, m2g_indices, m2g_weights
    )

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"\nModel parameters: {param_count:,}")

    # Forward pass
    print(f"\nRunning forward pass...")
    output = forward_fn.apply(
        params,
        rng,
        grid_input,
        mesh_graph,
        g2m_indices,
        g2m_weights,
        m2g_indices,
        m2g_weights,
    )

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({num_grid}, 2)")

    assert output.shape == (num_grid, 2), "Output shape mismatch!"

    # Test autoregressive rollout
    print(f"\nTesting autoregressive rollout (4 steps)...")
    predictions = autoregressive_rollout(
        forward_fn.apply,
        params,
        rng,
        grid_input,
        num_steps=4,
        mesh_graph=mesh_graph,
        g2m_indices=g2m_indices,
        g2m_weights=g2m_weights,
        m2g_indices=m2g_indices,
        m2g_weights=m2g_weights,
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Expected: (4, {num_grid}, 2)")

    assert predictions.shape == (4, num_grid, 2), "Predictions shape mismatch!"

    print("ALL TESTS PASSED!!!!!!")


if __name__ == "__main__":
    test_model()
