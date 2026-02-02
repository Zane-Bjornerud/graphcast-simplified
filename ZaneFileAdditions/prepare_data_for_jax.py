# File is to convert preprocessed data to JAX-compatible format


import numpy as np
import jax.numpy as jnp
import jraph


def load_and_convert_data():
    # Load preprocessed data and convert to JAX format

    print("Loading preprocessed data...")

    # Load data
    data = np.load("preprocessed_data.npz")
    mesh_data = np.load("mesh_data.npz")
    mappings = np.load("grid_mesh_mappings.npz")

    # Convert to JAX arrays
    train_inputs = jnp.array(data["train_inputs"])
    train_targets = jnp.array(data["train_targets"])
    val_inputs = jnp.array(data["val_inputs"])
    val_targets = jnp.array(data["val_targets"])
    test_inputs = jnp.array(data["test_inputs"])
    test_targets = jnp.array(data["test_targets"])

    print(f"Train inputs: {train_inputs.shape}")
    print(f"Train targets: {train_targets.shape}")
    print(f"Val inputs: {val_inputs.shape}")
    print(f"Val targets: {val_targets.shape}")
    print(f"Test inputs: {test_inputs.shape}")
    print(f"Test targets: {test_targets.shape}")

    # Create mesh graph
    vertices = jnp.array(mesh_data["vertices"])
    edges_array = mesh_data["edges"]
    edge_features = jnp.array(mesh_data["edge_features"])

    # Build Jraph graph
    senders = jnp.array(edges_array[:, 0])
    receivers = jnp.array(edges_array[:, 1])

    mesh_graph = jraph.GraphsTuple(
        nodes=vertices,  # [162, 3] xyz coordinates
        edges=edge_features,  # [480, 1] edge distances
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([vertices.shape[0]]),
        n_edge=jnp.array([edges_array.shape[0]]),
        globals=None,
    )

    print(
        f"Mesh graph: {mesh_graph.nodes.shape[0]} nodes, {mesh_graph.edges.shape[0]} edges"
    )

    # Load mappings
    g2m_indices = jnp.array(mappings["g2m_indices"])
    g2m_weights = jnp.array(mappings["g2m_weights"])
    m2g_indices = jnp.array(mappings["m2g_indices"])
    m2g_weights = jnp.array(mappings["m2g_weights"])

    print(f"Grid to Mesh mapping: {g2m_indices.shape}")
    print(f"Mesh to Grid mapping: {m2g_indices.shape}")

    return {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "val_inputs": val_inputs,
        "val_targets": val_targets,
        "test_inputs": test_inputs,
        "test_targets": test_targets,
        "mesh_graph": mesh_graph,
        "g2m_indices": g2m_indices,
        "g2m_weights": g2m_weights,
        "m2g_indices": m2g_indices,
        "m2g_weights": m2g_weights,
    }


if __name__ == "__main__":
    data_dict = load_and_convert_data()
    print("\nData ready for JAX")
