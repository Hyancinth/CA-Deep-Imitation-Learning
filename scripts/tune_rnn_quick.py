from pathlib import Path
from datetime import datetime
import itertools
import os

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
    plt.tight_layout()
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
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_bar_mean(df, x_col, y_col, save_path, title, rotate_xticks=False):
    grouped = df.groupby(x_col)[y_col].mean().sort_index()

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, axis="y")
    if rotate_xticks:
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_scatter_epochs_vs_mse(df, save_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["num_epochs"], df["test_mse"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Test MSE")
    plt.title("Epochs vs Test MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_scatter_lr_vs_mse(df, save_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["learning_rate"], df["test_mse"])
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test MSE")
    plt.title("Learning Rate vs Test MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_heatmap_hidden_vs_layers(df, save_path):
    pivot = df.pivot_table(
        values="test_mse",
        index="hidden_size",
        columns="num_layers",
        aggfunc="mean"
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Mean Test MSE")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Number of Layers")
    plt.ylabel("Hidden Size")
    plt.title("Hidden Size vs Number of Layers")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{value:.4f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_top_models(df, save_path, top_n=8):
    top_df = df.nsmallest(top_n, "test_mse").copy()

    labels = [
        f"hs={row.hidden_size}, L={row.num_layers}\n{row.feature_set}, ep={row.num_epochs}"
        for _, row in top_df.iterrows()
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_df)), top_df["test_mse"])
    plt.xticks(range(len(top_df)), labels, rotation=30, ha="right")
    plt.ylabel("Test MSE")
    plt.title(f"Top {top_n} Quick-Tuning Runs")
    plt.grid(True, axis="y")
    plt.tight_layout()
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

    training_file_path = repo_root / "model" / "data" / "data_322_01_100.h5"
    # If your file is in top-level data/, change the line above to:
    # training_file_path = repo_root / "data" / "data_322_01_100.h5"

    dataset_name = training_file_path.stem  # e.g. "data_322_01_100"
    output_dir = repo_root / "model" / "rnn_quick_outputs" / dataset_name
    model_dir  = repo_root / "model" / "trained_models"    / dataset_name
    scaler_dir = repo_root / "model" / "scalers"           / dataset_name

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

    results_csv_path = output_dir / f"rnn_quick_results_{timestamp}.csv"
    results_df.to_csv(results_csv_path, index=False)

    best_result_df = pd.DataFrame([best_result])
    best_result_csv_path = output_dir / f"best_rnn_quick_result_{timestamp}.csv"
    best_result_df.to_csv(best_result_csv_path, index=False)

    # Hyperparameter visualization plots
    plot_bar_mean(
        results_df,
        x_col="hidden_size",
        y_col="test_mse",
        save_path=output_dir / f"quick_hidden_size_vs_mse_{timestamp}.png",
        title="Average Test MSE by Hidden Size",
    )

    plot_bar_mean(
        results_df,
        x_col="num_layers",
        y_col="test_mse",
        save_path=output_dir / f"quick_num_layers_vs_mse_{timestamp}.png",
        title="Average Test MSE by Number of Layers",
    )

    plot_bar_mean(
        results_df,
        x_col="feature_set",
        y_col="test_mse",
        save_path=output_dir / f"quick_feature_set_vs_mse_{timestamp}.png",
        title="Average Test MSE by Feature Set",
        rotate_xticks=True,
    )

    plot_bar_mean(
        results_df,
        x_col="num_epochs",
        y_col="test_mse",
        save_path=output_dir / f"quick_epochs_vs_mse_{timestamp}.png",
        title="Average Test MSE by Number of Epochs",
    )

    plot_bar_mean(
        results_df,
        x_col="dropout",
        y_col="test_mse",
        save_path=output_dir / f"quick_dropout_vs_mse_{timestamp}.png",
        title="Average Test MSE by Dropout",
    )

    plot_bar_mean(
        results_df,
        x_col="learning_rate",
        y_col="test_mse",
        save_path=output_dir / f"quick_learning_rate_vs_mse_bar_{timestamp}.png",
        title="Average Test MSE by Learning Rate",
    )

    plot_scatter_epochs_vs_mse(
        results_df,
        save_path=output_dir / f"quick_epochs_vs_mse_scatter_{timestamp}.png",
    )

    plot_scatter_lr_vs_mse(
        results_df,
        save_path=output_dir / f"quick_learning_rate_vs_mse_scatter_{timestamp}.png",
    )

    plot_heatmap_hidden_vs_layers(
        results_df,
        save_path=output_dir / f"quick_hidden_vs_layers_heatmap_{timestamp}.png",
    )

    plot_top_models(
        results_df,
        save_path=output_dir / f"quick_top_models_{timestamp}.png",
        top_n=8,
    )

    print("\nTop results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved full results CSV to: {results_csv_path}")
    print(f"Saved best result CSV to: {best_result_csv_path}")

    if best_artifacts is not None:
        best_model_path = model_dir / f"best_rnn_quick_{timestamp}.pt"
        best_scaler_path = scaler_dir / f"best_rnn_quick_scaler_{timestamp}.pkl"
        best_loss_plot_path = output_dir / f"best_rnn_quick_loss_{timestamp}.png"
        best_pred_plot_path = output_dir / f"best_rnn_quick_prediction_{timestamp}.png"

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

        summary_txt_path = output_dir / f"best_rnn_quick_summary_{timestamp}.txt"
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            for k, v in best_result.items():
                f.write(f"{k}: {v}\n")

        print("\nBest result:")
        print(best_result)
        print(f"Saved best model to: {best_model_path}")
        print(f"Saved best scaler to: {best_scaler_path}")
        print(f"Saved best loss plot to: {best_loss_plot_path}")
        print(f"Saved best prediction plot to: {best_pred_plot_path}")
        print(f"Saved best summary text to: {summary_txt_path}")

    print("\nSaved hyperparameter tuning visualizations:")
    print(output_dir / f"quick_hidden_size_vs_mse_{timestamp}.png")
    print(output_dir / f"quick_num_layers_vs_mse_{timestamp}.png")
    print(output_dir / f"quick_feature_set_vs_mse_{timestamp}.png")
    print(output_dir / f"quick_epochs_vs_mse_{timestamp}.png")
    print(output_dir / f"quick_dropout_vs_mse_{timestamp}.png")
    print(output_dir / f"quick_learning_rate_vs_mse_bar_{timestamp}.png")
    print(output_dir / f"quick_epochs_vs_mse_scatter_{timestamp}.png")
    print(output_dir / f"quick_learning_rate_vs_mse_scatter_{timestamp}.png")
    print(output_dir / f"quick_hidden_vs_layers_heatmap_{timestamp}.png")
    print(output_dir / f"quick_top_models_{timestamp}.png")


if __name__ == "__main__":
    main()