"""
Run the trained RNN model autoregressively on hidden test datasets and save predictions.

Performs step-by-step rollout with proper hidden-state propagation so the RNN's
memory is carried between time steps (matching how it would behave in deployment).

Usage
-----
Set MODEL_PATH, SCALER_PATH, EXCLUDE_COLUMNS, and HIDDEN_TEST_FILES in the
configuration block near the bottom, then run:

    python -m scripts.run_rnn_hidden_test

Outputs are written to analysis/model_predictions/ in the same H5 format used
by visualize_comparison.py.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from data.load_data import load_data_from_file
from data.write_data import write_data_to_file
from model.rnn_data import DEFAULT_X_COLUMNS
from model.rnn_model import RNNPolicy
from utils.utils import dist_to_links, fk


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_rnn_feature_vector(
    theta: np.ndarray,
    target: np.ndarray,
    obstacle: np.ndarray,
    u_prev: np.ndarray,
    a: list,
    exclude_columns: list[str] | None = None,
) -> np.ndarray:
    """
    Build a feature vector for one time step, matching the 14-column schema
    used during RNN training (DEFAULT_X_COLUMNS in rnn_data.py).

    Parameters
    ----------
    theta        : [theta1, theta2]
    target       : [target_x, target_y]
    obstacle     : [obstacle_x, obstacle_y]
    u_prev       : [u1_prev, u2_prev]
    a            : link lengths [a1, a2]
    exclude_columns : columns to drop (must match what was used at training time)
    """
    if exclude_columns is None:
        exclude_columns = []

    theta1, theta2 = float(theta[0]), float(theta[1])
    x1, y1, x2, y2 = fk(theta, a)

    ee_dist_to_target = float(np.linalg.norm(np.array([x2, y2]) - target))
    ee_dist_to_obstacle = float(np.linalg.norm(np.array([x2, y2]) - obstacle))
    dist_links = dist_to_links(obstacle, theta, a)
    ee_dx_target = float(target[0] - x2)
    ee_dy_target = float(target[1] - y2)

    feature_dict = {
        "theta1":                  theta1,
        "theta2":                  theta2,
        "target_x":                float(target[0]),
        "target_y":                float(target[1]),
        "obstacle_x":              float(obstacle[0]),
        "obstacle_y":              float(obstacle[1]),
        "ee_dist_to_target":       ee_dist_to_target,
        "ee_dist_to_obstacle":     ee_dist_to_obstacle,
        "min_dist_obstacle_link_1": float(dist_links[0]),
        "min_dist_obstacle_link_2": float(dist_links[1]),
        "u1_prev":                 float(u_prev[0]),
        "u2_prev":                 float(u_prev[1]),
        "ee_dx_target":            ee_dx_target,
        "ee_dy_target":            ee_dy_target,
    }

    return np.array(
        [feature_dict[col] for col in DEFAULT_X_COLUMNS if col not in exclude_columns],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------

def run_rnn_rollout(
    model: RNNPolicy,
    scaler,
    theta0: np.ndarray,
    target: np.ndarray,
    obstacle: np.ndarray,
    a: list,
    exclude_columns: list[str] | None = None,
    num_steps: int = 100,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out the RNN model step by step, propagating the hidden state between
    steps so the network retains temporal memory across the full trajectory.

    Returns
    -------
    ee_trajectory      : (num_steps, 2)
    joint1_trajectory  : (num_steps, 2)
    theta_trajectory   : (num_steps, 2)
    u_trajectory       : (num_steps, 2)
    """
    if exclude_columns is None:
        exclude_columns = []

    device = next(model.parameters()).device
    model.eval()

    theta = np.array(theta0, dtype=np.float64)
    u_prev = np.zeros(2, dtype=np.float64)
    h = None  # RNN hidden state — None uses the default zero initialisation

    theta_trajectory = []
    ee_trajectory = []
    joint1_trajectory = []
    u_trajectory = []

    with torch.no_grad():
        for step in range(num_steps):
            # Build and scale the feature vector for this step
            x = build_rnn_feature_vector(theta, target, obstacle, u_prev, a, exclude_columns)
            x_scaled = scaler.transform(x.reshape(1, -1))          # (1, F)
            # Shape for RNN: (batch=1, seq_len=1, features)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward through RNN core with persistent hidden state
            rnn_out, h = model.rnn(x_tensor, h)   # rnn_out: (1,1,H), h: (L,1,H)
            u_tensor = model.head(rnn_out)         # (1, 1, 2)
            u = u_tensor[0, 0, :].cpu().numpy().astype(np.float64)
            u = np.clip(u, -3.0, 3.0)

            # Euler integration: theta_{t+1} = theta_t + u * dt
            theta = theta + u * dt
            u_prev = u.copy()

            j1_pos = np.array(fk(theta, a)[:2])
            ee_pos = np.array(fk(theta, a)[2:4])

            joint1_trajectory.append(j1_pos)
            ee_trajectory.append(ee_pos)
            theta_trajectory.append(theta.copy())
            u_trajectory.append(u.copy())

            print(
                f"  Step {step + 1:>3}/{num_steps} | "
                f"theta=[{theta[0]:.3f}, {theta[1]:.3f}] | "
                f"u=[{u[0]:.3f}, {u[1]:.3f}] | "
                f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}]"
            )

    return (
        np.array(ee_trajectory),
        np.array(joint1_trajectory),
        np.array(theta_trajectory),
        np.array(u_trajectory),
    )


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def run_on_hidden_file(
    model: RNNPolicy,
    scaler,
    hidden_file_path: str,
    training_data_name: str,
    exclude_columns: list[str],
    a: list,
    run_index: int = 0,
) -> str:
    """
    Run the RNN on one hidden test file (single run) and write the predictions
    to analysis/model_predictions/ in the expected H5 format.

    Returns the filename of the saved predictions.
    """
    hidden_file_name = Path(hidden_file_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"Hidden test file : {hidden_file_name}")
    print(f"Run index        : {run_index}")
    print(f"{'='*60}")

    data = load_data_from_file(hidden_file_path)
    run_key = f"run_{run_index}"

    if run_key not in data:
        raise KeyError(f"'{run_key}' not found in {hidden_file_path}")

    run_data = data[run_key]
    theta0    = np.array([run_data["theta1"][0],    run_data["theta2"][0]])
    target    = np.array([run_data["target_x"][0],  run_data["target_y"][0]])
    obstacle  = np.array([run_data["obstacle_x"][0], run_data["obstacle_y"][0]])

    print(f"theta0   = {theta0}")
    print(f"target   = {target}")
    print(f"obstacle = {obstacle}")

    ee_traj, j1_traj, theta_traj, u_traj = run_rnn_rollout(
        model, scaler, theta0, target, obstacle, a,
        exclude_columns=exclude_columns,
    )

    excl_str = "_".join(exclude_columns) if exclude_columns else "all_features"
    filename = (
        f"rnn_predictions_{training_data_name}_{hidden_file_name}"
        f"_run{run_index}_{timestamp}_excl_{excl_str}.h5"
    )

    data_to_save = {
        "run_number":            run_index,
        "theta1":                theta_traj[:, 0],
        "theta2":                theta_traj[:, 1],
        "u1":                    u_traj[:, 0],
        "u2":                    u_traj[:, 1],
        "ee_x":                  ee_traj[:, 0],
        "ee_y":                  ee_traj[:, 1],
        "joint1_x":              j1_traj[:, 0],
        "joint1_y":              j1_traj[:, 1],
        "target_x":              np.array([target[0]]),
        "target_y":              np.array([target[1]]),
        "obstacle_x":            np.array([obstacle[0]]),
        "obstacle_y":            np.array([obstacle[1]]),
        "hidden_data_file_path": hidden_file_path,
    }

    write_data_to_file(data_to_save, filename=filename, type="model_prediction")
    print(f"\nPredictions saved → analysis/model_predictions/{filename}")
    return filename


# ---------------------------------------------------------------------------
# Configuration & entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # USER CONFIGURATION
    # ------------------------------------------------------------------
    # Paths to the trained RNN model and its feature scaler.
    # After running tune_rnn_quick.py or run_rnn_model.py these files will be
    # under model/trained_models/ and model/scalers/ respectively.
    # Update the filenames below to point to your chosen best model.
    # ------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[1]

    MODEL_PATH  = str(repo_root / "model" / "trained_models" / "best_rnn_quick_<TIMESTAMP>.pt")
    SCALER_PATH = str(repo_root / "model" / "scalers"        / "best_rnn_quick_scaler_<TIMESTAMP>.pkl")

    # Must match the exclude_columns used when training the model
    EXCLUDE_COLUMNS: list[str] = []
    # e.g. EXCLUDE_COLUMNS = ["u1_prev", "u2_prev"]

    # Training dataset name (used only for output file naming)
    TRAINING_DATA_NAME = "data_322_01_100"

    # Hidden test files to evaluate
    HIDDEN_TEST_FILES = [
        str(repo_root / "model" / "hidden_test_data" / "hidden_test_data_2.h5"),
        str(repo_root / "model" / "hidden_test_data" / "hidden_test_data_5.h5"),
        str(repo_root / "model" / "hidden_test_data" / "hidden_test_data_6.h5"),
    ]

    # Run index to evaluate (0 = first scenario in each file)
    RUN_INDEX = 0

    # Robot link lengths
    A = [1.0, 1.0]
    # ------------------------------------------------------------------

    # Determine input size from the feature set
    input_size = len(DEFAULT_X_COLUMNS) - len(EXCLUDE_COLUMNS)

    print(f"Loading model  : {MODEL_PATH}")
    print(f"Loading scaler : {SCALER_PATH}")
    print(f"Input features : {input_size}")
    print(f"Excluded cols  : {EXCLUDE_COLUMNS or 'none'}")

    if "<TIMESTAMP>" in MODEL_PATH or "<TIMESTAMP>" in SCALER_PATH:
        raise ValueError(
            "Please update MODEL_PATH and SCALER_PATH with actual filenames "
            "from your latest training run before executing this script."
        )

    # Load model
    model = RNNPolicy(input_size=input_size, hidden_size=64, num_layers=2, output_size=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Run on each hidden test file
    prediction_files = {}
    for hidden_path in HIDDEN_TEST_FILES:
        pred_filename = run_on_hidden_file(
            model=model,
            scaler=scaler,
            hidden_file_path=hidden_path,
            training_data_name=TRAINING_DATA_NAME,
            exclude_columns=EXCLUDE_COLUMNS,
            a=A,
            run_index=RUN_INDEX,
        )
        prediction_files[Path(hidden_path).stem] = pred_filename

    print("\n" + "="*60)
    print("All predictions complete. Saved files:")
    for hidden_name, pred_file in prediction_files.items():
        print(f"  {hidden_name:30s} -> {pred_file}")
    print("="*60)
    print("\nNext step: run scripts/visualize_rnn_results.py to generate comparison plots.")
