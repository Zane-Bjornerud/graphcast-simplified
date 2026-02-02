# Training script for simplified GraphCast
# Based on autoregressive.py

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from tqdm import tqdm
import pickle
import os

from simple_graphcast import build_model_forward, autoregressive_rollout


def compute_loss(
    predictions: jnp.ndarray,  # [num_steps, num_grid, 2]
    targets: jnp.ndarray,  # [num_steps, num_grid, 2]
    per_variable_weights: jnp.ndarray = None,
) -> jnp.ndarray:

    # Compute MSE loss for predictions
    # Based on GraphCast losses.py
    # MSE per timestep and variable
    squared_error = jnp.square(predictions - targets)

    # Apply per-variable weights if provided
    if per_variable_weights is not None:
        squared_error = squared_error * per_variable_weights

    # Average over all dimensions
    loss = jnp.mean(squared_error)

    return loss


def compute_weighted_loss(
    predictions: jnp.ndarray, targets: jnp.ndarray, time_weights: jnp.ndarray = None
) -> jnp.ndarray:

    # Compute loss with time-dependent weights
    # GraphCast uses higher weights for later timesteps to encourage better long-term predictions

    if time_weights is None:
        # Default: equal weights
        time_weights = jnp.ones(predictions.shape[0])

    # Compute MSE per timestep
    timestep_losses = []
    for t in range(predictions.shape[0]):
        mse = jnp.mean(jnp.square(predictions[t] - targets[t]))
        timestep_losses.append(mse * time_weights[t])

    # Weighted average
    total_loss = jnp.sum(jnp.array(timestep_losses)) / jnp.sum(time_weights)

    return total_loss


def create_train_step(forward_fn, optimizer, num_ar_steps, grid_shape=(32, 64)):
    # Create the training step function
    # Based on GraphCast training loop pattern

    num_lat, num_lon = grid_shape
    num_grid = num_lat * num_lon  # 32 * 64 = 2048

    def loss_fn(
        params,
        rng,
        batch,
        mesh_graph,
        g2m_indices,
        g2m_weights,
        m2g_indices,
        m2g_weights,
    ):
        # Compute loss for one batch

        # Args:
        #     batch: dict with 'inputs' and 'targets'
        #         inputs: [batch_size, 32, 64, 2]
        #         targets: [batch_size, num_ar_steps, 32, 64, 2]

        inputs = batch["inputs"]
        targets = batch["targets"]

        batch_size = inputs.shape[0]

        # Process each sample in batch
        total_loss = 0.0
        for i in range(batch_size):
            # Flatten input for model: [32, 64, 2] to [2048, 2]
            input_flat = inputs[i].reshape(num_grid, 2)

            # Autoregressive rollout (model works with flat grids)
            predictions_flat = autoregressive_rollout(
                forward_fn,
                params,
                rng,
                input_flat,
                num_ar_steps,
                mesh_graph,
                g2m_indices,
                g2m_weights,
                m2g_indices,
                m2g_weights,
            )
            # predictions_flat shape: [num_ar_steps, 2048, 2]

            # Reshape predictions back to spatial grid: [4, 2048, 2] to [4, 32, 64, 2]
            predictions_grid = predictions_flat.reshape(
                num_ar_steps, num_lat, num_lon, 2
            )

            # Compute loss (both have spatial structure)
            # predictions_grid: [4, 32, 64, 2]
            # targets[i]: [4, 32, 64, 2]
            sample_loss = compute_loss(predictions_grid, targets[i])
            total_loss += sample_loss

        # Average over batch
        return total_loss / batch_size

    @jax.jit
    def train_step(
        params,
        opt_state,
        rng,
        batch,
        mesh_graph,
        g2m_indices,
        g2m_weights,
        m2g_indices,
        m2g_weights,
    ):
        # Single training step with gradient update
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(
            params,
            rng,
            batch,
            mesh_graph,
            g2m_indices,
            g2m_weights,
            m2g_indices,
            m2g_weights,
        )

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    return train_step


def create_batches(inputs, targets, batch_size):
    # Create batches from data
    # Returns list of batches

    num_sequences = inputs.shape[0]
    num_batches = num_sequences // batch_size

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch = {
            "inputs": inputs[start_idx:end_idx],
            "targets": targets[start_idx:end_idx],
        }
        batches.append(batch)

    return batches


def train(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    mesh_graph,
    g2m_indices,
    g2m_weights,
    m2g_indices,
    m2g_weights,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_ar_steps: int = 4,
    grid_shape: tuple = (32, 64),
    checkpoint_dir: str = "checkpoints",
):

    # Main training loop
    # Based on GraphCast training procedure

    print("TRAINING GRAPHCAST")
    print(f"Training samples: {train_inputs.shape[0]:,}")
    print(f"Validation samples: {val_inputs.shape[0]:,}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Autoregressive steps: {num_ar_steps}")
    print(f"Grid shape: {grid_shape}")
    print(f"Epochs: {num_epochs}")

    # Convert to JAX arrays
    train_inputs_jax = jnp.array(train_inputs)
    train_targets_jax = jnp.array(train_targets)
    val_inputs_jax = jnp.array(val_inputs)
    val_targets_jax = jnp.array(val_targets)

    # Build model
    forward_fn = build_model_forward()
    forward_fn = hk.transform(forward_fn)

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    # Flatten first input for initialization: [32, 64, 2] to [2048, 2]
    num_lat, num_lon = grid_shape
    num_grid = num_lat * num_lon
    init_input = train_inputs_jax[0].reshape(num_grid, 2)

    params = forward_fn.init(
        init_rng,
        init_input,  # Flattened input
        mesh_graph,
        g2m_indices,
        g2m_weights,
        m2g_indices,
        m2g_weights,
    )

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {param_count:,}\n")

    # Create optimizer (Adam with learning rate schedule)
    # Based on GraphCast autoregressive.py
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Create training step (pass num_ar_steps and grid_shape to closure)
    train_step_fn = create_train_step(
        forward_fn.apply, optimizer, num_ar_steps, grid_shape
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Create batches (shuffle each epoch)
        rng, shuffle_rng = jax.random.split(rng)
        shuffle_idx = jax.random.permutation(shuffle_rng, train_inputs_jax.shape[0])

        train_inputs_shuffled = train_inputs_jax[shuffle_idx]
        train_targets_shuffled = train_targets_jax[shuffle_idx]

        batches = create_batches(
            train_inputs_shuffled, train_targets_shuffled, batch_size
        )

        # Training
        train_losses = []
        for batch in tqdm(batches, desc="Training"):
            rng, step_rng = jax.random.split(rng)

            params, opt_state, loss = train_step_fn(
                params,
                opt_state,
                step_rng,
                batch,
                mesh_graph,
                g2m_indices,
                g2m_weights,
                m2g_indices,
                m2g_weights,
            )

            train_losses.append(float(loss))

        avg_train_loss = np.mean(train_losses)

        # Validation
        val_batches = create_batches(val_inputs_jax, val_targets_jax, batch_size)
        val_losses = []

        for batch in val_batches:
            rng, val_rng = jax.random.split(rng)

            # Compute validation loss (no gradient update)
            val_loss = 0.0
            for i in range(batch["inputs"].shape[0]):
                # Flatten input for model: [32, 64, 2] to [2048, 2]
                input_flat = batch["inputs"][i].reshape(num_grid, 2)

                # Predict (flat output)
                predictions_flat = autoregressive_rollout(
                    forward_fn.apply,
                    params,
                    val_rng,
                    input_flat,  # [2048, 2]
                    num_ar_steps,
                    mesh_graph,
                    g2m_indices,
                    g2m_weights,
                    m2g_indices,
                    m2g_weights,
                )
                # predictions_flat shape: [4, 2048, 2]

                # Reshape to spatial grid: [4, 2048, 2] to [4, 32, 64, 2]
                predictions_grid = predictions_flat.reshape(
                    num_ar_steps, num_lat, num_lon, 2
                )

                # Compute loss (spatial structure preserved)
                # predictions_grid: [4, 32, 64, 2]
                # batch['targets'][i]: [4, 32, 64, 2]
                val_loss += float(compute_loss(predictions_grid, batch["targets"][i]))

            val_losses.append(val_loss / batch["inputs"].shape[0])

        avg_val_loss = np.mean(val_losses)

        # Print progress
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss! Saving checkpoint")

            with open(f"{checkpoint_dir}/best_params.pkl", "wb") as f:
                pickle.dump(params, f)

    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return params


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading data...")

    from prepare_data_for_jax import load_and_convert_data

    data_dict = load_and_convert_data()

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Train
    params = train(
        train_inputs=data_dict["train_inputs"],
        train_targets=data_dict["train_targets"],
        val_inputs=data_dict["val_inputs"][:1000],  # Use subset for faster validation
        val_targets=data_dict["val_targets"][:1000],
        mesh_graph=data_dict["mesh_graph"],
        g2m_indices=data_dict["g2m_indices"],
        g2m_weights=data_dict["g2m_weights"],
        m2g_indices=data_dict["m2g_indices"],
        m2g_weights=data_dict["m2g_weights"],
        num_epochs=10,
        batch_size=4,
        learning_rate=1e-4,
        num_ar_steps=4,
        grid_shape=(32, 64),
    )
