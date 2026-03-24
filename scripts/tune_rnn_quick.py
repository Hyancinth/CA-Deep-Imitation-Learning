import os
from pathlib import Path
from datetime import datetime
import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model.rnn_data import (
    load_h5_runs,
    build_sequences_from_runs,
    split_runs,
    scale_sequence_features,
)
from model.rnn_model import RNNPolicy
from model.train_test_rnn import (
    create_sequence_data_loaders,
    train_sequence_model,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evaluate_on_test_set(model, x_test, y_test, device):
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()
    mse = np.mean((preds - y_test) ** 2)
    mae = np.mean(np.abs(preds - y_test))
    return preds, mse, mae


def plot_losses(train_losses, test_losses, save_path=None, title="Loss Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_prediction_vs_truth(y_true, y_pred, save_path=None, title="Prediction vs Ground Truth"):
    t = np.arange(y_true.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_true[:, 0], label="True u1")
    plt.plot(t, y_pred[:, 0], "--", label="Pred u1")
    plt.plot(t, y_true[:, 1], label="True u2")
    plt.plot(t, y_pred[:, 1], "--", label="Pred u2")
    plt.xlabel("Time Step")
    plt.ylabel("Control Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_single_experiment(
    data,
    hidden_size,
    num_layers,
    num_epochs,
    learning_rate,
    dropout,
    feature_name,
    exclude_columns,
    batch_size=16,
    test_size=0.2,
    random_state=42,
):
    X, Y, used_x_columns, y_columns = build_sequences_from_runs(
        data,
        exclude_columns=exclude_columns,
    )

    X_train, X_test, Y_train, Y_test = split_runs(
        X, Y, test_size=test_size, random_state=random_state
    )

    X_train_scaled, X_test_scaled, scaler = scale_sequence_features(X_train, X_test)

    train_loader, test_loader = create_sequence_data_loaders(
        X_train_scaled,
        Y_train,
        X_test_scaled,
        Y_test,
        batch_size=batch_size,
    )

    model = RNNPolicy(
        input_size=X_train_scaled.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=Y_train.shape[2],
        dropout=dropout,
        nonlinearity="tanh",
        bidirectional=False,
    )

    model, train_losses, test_losses, device = train_sequence_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=1e-5,
    )

    preds, mse, mae = evaluate_on_test_set(model, X_test_scaled, Y_test, device)

    result = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "feature_set": feature_name,
        "exclude_columns": ", ".join(exclude_columns) if exclude_columns else "None",
        "best_epoch": int(np.argmin(test_losses) + 1),
        "best_test_loss": float(np.min(test_losses)),
        "final_train_loss": float(train_losses[-1]),
        "final_test_loss": float(test_losses[-1]),
        "test_mse": float(mse),
        "test_mae": float(mae),
    }

    artifacts = {
        "model": model,
        "scaler": scaler,
        "Y_test": Y_test,
        "preds": preds,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "used_x_columns": used_x_columns,
        "y_columns": y_columns,
    }

    return result, artifacts


def main():
    repo_root = Path(__file__).resolve().parents[1]
    training_file_path = repo_root / "model" / "data" / "data_320_01_100.h5"

    output_dir = repo_root / "model" / "rnn_quick_outputs"
    model_dir = repo_root / "model" / "trained_models"
    scaler_dir = repo_root / "model" / "scalers"

    ensure_dir(output_dir)
    ensure_dir(model_dir)
    ensure_dir(scaler_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    data = load_h5_runs(str(training_file_path))

    feature_sets = {
        "all_features": [],
        "no_prev_u": ["u1_prev", "u2_prev"],
    }

    architectures = [
        {"hidden_size": 32, "num_layers": 1},
        {"hidden_size": 64, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 2},
    ]

    training_settings = [
        {"num_epochs": 150, "learning_rate": 1e-3, "dropout": 0.1},
        {"num_epochs": 100, "learning_rate": 5e-4, "dropout": 0.0},
    ]

    combinations = list(itertools.product(architectures, feature_sets.items(), training_settings))
    print(f"Total quick experiments: {len(combinations)}")

    results = []
    best_mse = float("inf")
    best_result = None
    best_artifacts = None

    for i, (arch, feature_info, train_cfg) in enumerate(combinations, start=1):
        feature_name, exclude_columns = feature_info

        exp_name = (
            f"quick_hs{arch['hidden_size']}_layers{arch['num_layers']}_"
            f"{feature_name}_ep{train_cfg['num_epochs']}_"
            f"lr{train_cfg['learning_rate']}_do{train_cfg['dropout']}"
        )

        print(f"\n[{i}/{len(combinations)}] {exp_name}")

        result, artifacts = run_single_experiment(
            data=data,
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            num_epochs=train_cfg["num_epochs"],
            learning_rate=train_cfg["learning_rate"],
            dropout=train_cfg["dropout"],
            feature_name=feature_name,
            exclude_columns=exclude_columns,
            batch_size=16,
        )

        result["experiment_name"] = exp_name
        results.append(result)

        if i <= 4:
            plot_losses(
                artifacts["train_losses"],
                artifacts["test_losses"],
                save_path=output_dir / f"{exp_name}_loss.png",
                title=exp_name,
            )

        if result["test_mse"] < best_mse:
            best_mse = result["test_mse"]
            best_result = result
            best_artifacts = artifacts

    results_df = pd.DataFrame(results).sort_values(by="test_mse", ascending=True)
    csv_path = output_dir / f"rnn_quick_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    print("\nTop results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results CSV to: {csv_path}")

    if best_artifacts is not None:
        best_model_path = model_dir / f"best_rnn_quick_{timestamp}.pt"
        best_scaler_path = scaler_dir / f"best_rnn_quick_scaler_{timestamp}.pkl"
        best_loss_plot_path = output_dir / f"best_rnn_quick_loss_{timestamp}.png"
        best_pred_plot_path = output_dir / f"best_rnn_quick_prediction_{timestamp}.png"
        best_summary_path = output_dir / f"best_rnn_quick_summary_{timestamp}.txt"

        torch.save(best_artifacts["model"].state_dict(), best_model_path)
        joblib.dump(best_artifacts["scaler"], best_scaler_path)

        plot_losses(
            best_artifacts["train_losses"],
            best_artifacts["test_losses"],
            save_path=best_loss_plot_path,
            title="Best Quick RNN Loss Curve",
        )

        plot_prediction_vs_truth(
            best_artifacts["Y_test"][0],
            best_artifacts["preds"][0],
            save_path=best_pred_plot_path,
            title="Best Quick RNN Prediction vs Ground Truth",
        )

        with open(best_summary_path, "w", encoding="utf-8") as f:
            for k, v in best_result.items():
                f.write(f"{k}: {v}\n")

        print("\nBest result:")
        print(best_result)
        print(f"Saved best model to: {best_model_path}")
        print(f"Saved best scaler to: {best_scaler_path}")
        print(f"Saved best loss plot to: {best_loss_plot_path}")
        print(f"Saved best prediction plot to: {best_pred_plot_path}")
        print(f"Saved best summary to: {best_summary_path}")


if __name__ == "__main__":
    main()