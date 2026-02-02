import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pickle
import matplotlib.pyplot as plt

from simple_graphcast import build_model_forward, autoregressive_rollout


def load_best_params(checkpoint_path="checkpoints/best_params.pkl"):
    # Load best model parameters from checkpoint
    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
    print("Loaded best model parameters")
    return params


def persistence_baseline(inputs, num_ar_steps=4):
    # A good model SHOULD beat this but maybe this one is not good enough
    # Returns predictions: [num_samples, num_ar_steps, 32, 64, 2]

    # Repeat the input for each timestep
    predictions = jnp.stack([inputs] * num_ar_steps, axis=1)
    return predictions


def compute_rmse(predictions, targets):
    squared_error = jnp.square(predictions - targets)

    # RMSE per variable (average over samples, timesteps, lat, lon)
    mse_per_var = jnp.mean(squared_error, axis=(0, 1, 2, 3))
    rmse_per_var = jnp.sqrt(mse_per_var)

    # RMSE per timestep per variable
    mse_per_step = jnp.mean(squared_error, axis=(0, 2, 3))
    rmse_per_step = jnp.sqrt(mse_per_step)

    return rmse_per_var, rmse_per_step


def evaluate_model(
    params,
    test_inputs,
    test_targets,
    mesh_graph,
    g2m_indices,
    g2m_weights,
    m2g_indices,
    m2g_weights,
    num_ar_steps=4,
    grid_shape=(32, 64),
    num_samples=500,
):
    num_lat, num_lon = grid_shape
    num_grid = num_lat * num_lon

    # Build model
    forward_fn = build_model_forward()
    forward_fn = hk.transform(forward_fn)

    # Use subset for speed
    test_inputs_subset = jnp.array(test_inputs[:num_samples])
    test_targets_subset = jnp.array(test_targets[:num_samples])

    print(f"\nEvaluating on {num_samples} test samples...")

    # Get model predictions
    all_predictions = []
    rng = jax.random.PRNGKey(0)

    for i in range(num_samples):
        if i % 100 == 0:
            print(f"  Predicting sample {i}/{num_samples}...")

        # Flatten input: [32, 64, 2] to [2048, 2]
        input_flat = test_inputs_subset[i].reshape(num_grid, 2)

        # Predict
        predictions_flat = autoregressive_rollout(
            forward_fn.apply,
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
        # predictions_flat: [4, 2048, 2]

        # Reshape to grid: [4, 2048, 2] to [4, 32, 64, 2]
        predictions_grid = predictions_flat.reshape(num_ar_steps, num_lat, num_lon, 2)
        all_predictions.append(predictions_grid)

    # Stack all predictions: [num_samples, num_ar_steps, 32, 64, 2]
    model_predictions = jnp.stack(all_predictions, axis=0)

    # Get persistence baseline predictions
    persistence_predictions = persistence_baseline(test_inputs_subset, num_ar_steps)

    # Compute RMSE
    model_rmse_var, model_rmse_step = compute_rmse(
        model_predictions, test_targets_subset
    )
    persist_rmse_var, persist_rmse_step = compute_rmse(
        persistence_predictions, test_targets_subset
    )

    return {
        "model_predictions": model_predictions,
        "persistence_predictions": persistence_predictions,
        "targets": test_targets_subset,
        "inputs": test_inputs_subset,
        "model_rmse_var": model_rmse_var,
        "model_rmse_step": model_rmse_step,
        "persist_rmse_var": persist_rmse_var,
        "persist_rmse_step": persist_rmse_step,
    }


def print_results(results):
    var_names = ["Geopotential (Z500)", "Temperature (T2m)"]
    print("EVALUATION RESULTS")
    # Overall RMSE
    print("\nOverall RMSE per variable:")
    print(f"{'Variable':<25} {'Model':>10} {'Persistence':>12} {'Better?':>10}")

    for v in range(2):
        model_val = float(results["model_rmse_var"][v])
        persist_val = float(results["persist_rmse_var"][v])
        better = "YES" if model_val < persist_val else " NO"
        print(
            f"{var_names[v]:<25} {model_val:>10.6f} {persist_val:>12.6f} {better:>10}"
        )

    # RMSE per timestep
    print("\nRMSE per prediction timestep:")

    for v in range(2):
        print(f"\n  {var_names[v]}:")
        print(f"  {'Step':<8} {'Model':>10} {'Persistence':>12} {'Better?':>10}")
        for t in range(results["model_rmse_step"].shape[0]):
            model_val = float(results["model_rmse_step"][t, v])
            persist_val = float(results["persist_rmse_step"][t, v])
            better = "" if model_val < persist_val else ""
            print(f"  t+{t+1:<5} {model_val:>10.6f} {persist_val:>12.6f} {better:>10}")


def plot_results(results, save_path="evaluation_plots"):
    # Generate evaluation plots
    import os

    os.makedirs(save_path, exist_ok=True)

    var_names = ["Geopotential (Z500)", "Temperature (T2m)"]

    # RMSE over prediction timesteps plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    timesteps = np.arange(1, results["model_rmse_step"].shape[0] + 1)

    for v in range(2):
        ax = axes[v]

        model_rmse = np.array(results["model_rmse_step"][:, v])
        persist_rmse = np.array(results["persist_rmse_step"][:, v])

        ax.plot(
            timesteps, model_rmse, "b-o", linewidth=2, markersize=8, label="GraphCast"
        )
        ax.plot(
            timesteps,
            persist_rmse,
            "r--s",
            linewidth=2,
            markersize=8,
            label="Persistence",
        )

        ax.set_xlabel("Prediction Timestep", fontsize=12)
        ax.set_ylabel("RMSE (normalized)", fontsize=12)
        ax.set_title(f"{var_names[v]}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timesteps)

    plt.suptitle("RMSE vs Prediction Timestep", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_path}/rmse_vs_timestep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}/rmse_vs_timestep.png")

    # Example prediction vs target (Geopotential) plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    sample_idx = 0  # First test sample

    for t in range(4):
        # Model prediction
        ax = axes[0, t]
        im = ax.imshow(
            results["model_predictions"][sample_idx, t, :, :, 0],
            cmap="RdBu_r",
            aspect="auto",
        )
        ax.set_title(f"Model t+{t+1}", fontsize=11)
        ax.set_ylabel("Latitude" if t == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Target
        ax = axes[1, t]
        im = ax.imshow(
            results["targets"][sample_idx, t, :, :, 0], cmap="RdBu_r", aspect="auto"
        )
        ax.set_title(f"Target t+{t+1}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if t == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Geopotential: Model Predictions vs Targets", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{save_path}/predictions_geopotential.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {save_path}/predictions_geopotential.png")

    # Example prediction vs target (Temperature) plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for t in range(4):
        # Model prediction
        ax = axes[0, t]
        im = ax.imshow(
            results["model_predictions"][sample_idx, t, :, :, 1],
            cmap="RdBu_r",
            aspect="auto",
        )
        ax.set_title(f"Model t+{t+1}", fontsize=11)
        ax.set_ylabel("Latitude" if t == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Target
        ax = axes[1, t]
        im = ax.imshow(
            results["targets"][sample_idx, t, :, :, 1], cmap="RdBu_r", aspect="auto"
        )
        ax.set_title(f"Target t+{t+1}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if t == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Temperature: Model Predictions vs Targets", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{save_path}/predictions_temperature.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {save_path}/predictions_temperature.png")

    # Error maps (Model - Target)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for t in range(4):
        # Geopotential error
        ax = axes[0, t]
        error = (
            results["model_predictions"][sample_idx, t, :, :, 0]
            - results["targets"][sample_idx, t, :, :, 0]
        )
        im = ax.imshow(error, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Z500 Error t+{t+1}", fontsize=11)
        ax.set_ylabel("Latitude" if t == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Temperature error
        ax = axes[1, t]
        error = (
            results["model_predictions"][sample_idx, t, :, :, 1]
            - results["targets"][sample_idx, t, :, :, 1]
        )
        im = ax.imshow(error, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"T2m Error t+{t+1}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude" if t == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Prediction Errors (Model - Target)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}/error_maps.png")


if __name__ == "__main__":
    from prepare_data_for_jax import load_and_convert_data

    # Load data
    print("Loading data...")
    data_dict = load_and_convert_data()

    # Load trained parameters
    params = load_best_params()

    # Evaluate
    results = evaluate_model(
        params=params,
        test_inputs=data_dict["test_inputs"],
        test_targets=data_dict["test_targets"],
        mesh_graph=data_dict["mesh_graph"],
        g2m_indices=data_dict["g2m_indices"],
        g2m_weights=data_dict["g2m_weights"],
        m2g_indices=data_dict["m2g_indices"],
        m2g_weights=data_dict["m2g_weights"],
        num_ar_steps=4,
        grid_shape=(32, 64),
        num_samples=500,
    )

    # Print results
    print_results(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results)

    print("EVALUATION COMPLETE")
