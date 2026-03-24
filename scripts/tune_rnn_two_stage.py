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


def plot_bar_from_df(df, x_col, y_col, save_path, title):
    grouped = df.groupby(x_col)[y_col].mean().sort_index()

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_heatmap(df, index_col, column_col, value_col, save_path, title):
    pivot = df.pivot_table(values=value_col, index=index_col, columns=column_col, aggfunc="mean")

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(column_col)
    plt.ylabel(index_col)
    plt.title(title)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{value:.4f}", ha="center", va="center")

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
    nonlinearity,
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
        nonlinearity=nonlinearity,
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
        "nonlinearity": nonlinearity,
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
    # If needed, change to:
    # training_file_path = repo_root / "data" / "data_320_01_100.h5"

    output_dir = repo_root / "model" / "rnn_two_stage_outputs"
    model_dir = repo_root / "model" / "trained_models"
    scaler_dir = repo_root / "model" / "scalers"

    ensure_dir(output_dir)
    ensure_dir(model_dir)
    ensure_dir(scaler_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    data = load_h5_runs(str(training_file_path))

    # ============================================================
    # STAGE 1: architecture search
    # ============================================================
    print("\n" + "=" * 80)
    print("STAGE 1: ARCHITECTURE SEARCH")
    print("=" * 80)

    stage1_hidden_sizes = [32, 64, 128]
    stage1_num_layers = [1, 2, 3]
    stage1_nonlinearities = ["tanh", "relu"]

    feature_sets = {
        "all_features": [],
        "no_prev_u": ["u1_prev", "u2_prev"],
        "no_prev_u_no_dxdy": ["u1_prev", "u2_prev", "ee_dx_target", "ee_dy_target"],
    }

    # fixed settings for stage 1
    stage1_num_epochs = 150
    stage1_learning_rate = 1e-3
    stage1_dropout = 0.1
    batch_size = 16

    stage1_results = []
    stage1_best_mse = float("inf")
    stage1_best_config = None
    stage1_best_artifacts = None

    stage1_combinations = list(itertools.product(
        stage1_hidden_sizes,
        stage1_num_layers,
        feature_sets.items(),
        stage1_nonlinearities,
    ))

    print(f"Stage 1 experiments: {len(stage1_combinations)}")

    for i, (hidden_size, num_layers, feature_info, nonlinearity) in enumerate(stage1_combinations, start=1):
        feature_name, exclude_columns = feature_info

        exp_name = f"stage1_hs{hidden_size}_layers{num_layers}_{feature_name}_{nonlinearity}"
        print(f"\n[{i}/{len(stage1_combinations)}] {exp_name}")

        result, artifacts = run_single_experiment(
            data=data,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_epochs=stage1_num_epochs,
            learning_rate=stage1_learning_rate,
            dropout=stage1_dropout,
            feature_name=feature_name,
            exclude_columns=exclude_columns,
            nonlinearity=nonlinearity,
            batch_size=batch_size,
        )

        result["experiment_name"] = exp_name
        stage1_results.append(result)

        loss_plot_path = output_dir / f"{exp_name}_loss.png"
        plot_losses(
            artifacts["train_losses"],
            artifacts["test_losses"],
            save_path=loss_plot_path,
            title=exp_name,
        )

        if result["test_mse"] < stage1_best_mse:
            stage1_best_mse = result["test_mse"]
            stage1_best_config = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "feature_name": feature_name,
                "exclude_columns": exclude_columns,
                "nonlinearity": nonlinearity,
            }
            stage1_best_artifacts = artifacts

    stage1_df = pd.DataFrame(stage1_results).sort_values(by="test_mse", ascending=True)
    stage1_csv = output_dir / f"stage1_results_{timestamp}.csv"
    stage1_df.to_csv(stage1_csv, index=False)

    print("\nTop Stage 1 results:")
    print(stage1_df.head(10).to_string(index=False))
    print(f"\nSaved Stage 1 CSV to: {stage1_csv}")
    print("\nBest Stage 1 config:")
    print(stage1_best_config)

    plot_bar_from_df(
        stage1_df,
        x_col="hidden_size",
        y_col="test_mse",
        save_path=output_dir / f"stage1_hidden_size_{timestamp}.png",
        title="Stage 1: Average Test MSE by Hidden Size",
    )

    plot_bar_from_df(
        stage1_df,
        x_col="num_layers",
        y_col="test_mse",
        save_path=output_dir / f"stage1_num_layers_{timestamp}.png",
        title="Stage 1: Average Test MSE by Number of Layers",
    )

    plot_bar_from_df(
        stage1_df,
        x_col="nonlinearity",
        y_col="test_mse",
        save_path=output_dir / f"stage1_nonlinearity_{timestamp}.png",
        title="Stage 1: Average Test MSE by Nonlinearity",
    )

    plot_bar_from_df(
        stage1_df,
        x_col="feature_set",
        y_col="test_mse",
        save_path=output_dir / f"stage1_feature_set_{timestamp}.png",
        title="Stage 1: Average Test MSE by Feature Set",
    )

    plot_heatmap(
        stage1_df,
        index_col="hidden_size",
        column_col="num_layers",
        value_col="test_mse",
        save_path=output_dir / f"stage1_heatmap_hidden_vs_layers_{timestamp}.png",
        title="Stage 1: Mean Test MSE (Hidden Size vs Num Layers)",
    )

    # save best stage 1 prediction plot
    if stage1_best_artifacts is not None:
        y_true_sample = stage1_best_artifacts["Y_test"][0]
        y_pred_sample = stage1_best_artifacts["preds"][0]
        plot_prediction_vs_truth(
            y_true_sample,
            y_pred_sample,
            save_path=output_dir / f"stage1_best_prediction_{timestamp}.png",
            title="Stage 1 Best Model: Prediction vs Ground Truth",
        )

    # ============================================================
    # STAGE 2: training hyperparameter search
    # ============================================================
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING HYPERPARAMETER SEARCH")
    print("=" * 80)

    best_hidden_size = stage1_best_config["hidden_size"]
    best_num_layers = stage1_best_config["num_layers"]
    best_feature_name = stage1_best_config["feature_name"]
    best_exclude_columns = stage1_best_config["exclude_columns"]
    best_nonlinearity = stage1_best_config["nonlinearity"]

    stage2_num_epochs = [100, 150, 200, 300]
    stage2_learning_rates = [1e-3, 5e-4, 2e-4]
    stage2_dropouts = [0.0, 0.1, 0.2]

    stage2_results = []
    stage2_best_mse = float("inf")
    stage2_best_result = None
    stage2_best_artifacts = None

    stage2_combinations = list(itertools.product(
        stage2_num_epochs,
        stage2_learning_rates,
        stage2_dropouts,
    ))

    print(f"Stage 2 experiments: {len(stage2_combinations)}")

    for i, (num_epochs, learning_rate, dropout) in enumerate(stage2_combinations, start=1):
        exp_name = (
            f"stage2_hs{best_hidden_size}_layers{best_num_layers}_"
            f"{best_feature_name}_{best_nonlinearity}_"
            f"ep{num_epochs}_lr{learning_rate}_do{dropout}"
        )
        print(f"\n[{i}/{len(stage2_combinations)}] {exp_name}")

        result, artifacts = run_single_experiment(
            data=data,
            hidden_size=best_hidden_size,
            num_layers=best_num_layers,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            dropout=dropout,
            feature_name=best_feature_name,
            exclude_columns=best_exclude_columns,
            nonlinearity=best_nonlinearity,
            batch_size=batch_size,
        )

        result["experiment_name"] = exp_name
        stage2_results.append(result)

        loss_plot_path = output_dir / f"{exp_name}_loss.png"
        plot_losses(
            artifacts["train_losses"],
            artifacts["test_losses"],
            save_path=loss_plot_path,
            title=exp_name,
        )

        if result["test_mse"] < stage2_best_mse:
            stage2_best_mse = result["test_mse"]
            stage2_best_result = result
            stage2_best_artifacts = artifacts

    stage2_df = pd.DataFrame(stage2_results).sort_values(by="test_mse", ascending=True)
    stage2_csv = output_dir / f"stage2_results_{timestamp}.csv"
    stage2_df.to_csv(stage2_csv, index=False)

    print("\nTop Stage 2 results:")
    print(stage2_df.head(10).to_string(index=False))
    print(f"\nSaved Stage 2 CSV to: {stage2_csv}")

    plot_bar_from_df(
        stage2_df,
        x_col="num_epochs",
        y_col="test_mse",
        save_path=output_dir / f"stage2_epochs_{timestamp}.png",
        title="Stage 2: Average Test MSE by Number of Epochs",
    )

    plot_bar_from_df(
        stage2_df,
        x_col="learning_rate",
        y_col="test_mse",
        save_path=output_dir / f"stage2_learning_rate_{timestamp}.png",
        title="Stage 2: Average Test MSE by Learning Rate",
    )

    plot_bar_from_df(
        stage2_df,
        x_col="dropout",
        y_col="test_mse",
        save_path=output_dir / f"stage2_dropout_{timestamp}.png",
        title="Stage 2: Average Test MSE by Dropout",
    )

    # ============================================================
    # Save final best model
    # ============================================================
    if stage2_best_artifacts is not None:
        best_model_path = model_dir / f"best_rnn_two_stage_{timestamp}.pt"
        best_scaler_path = scaler_dir / f"best_rnn_two_stage_scaler_{timestamp}.pkl"
        best_pred_plot_path = output_dir / f"best_rnn_two_stage_prediction_{timestamp}.png"
        best_loss_plot_path = output_dir / f"best_rnn_two_stage_loss_{timestamp}.png"

        torch.save(stage2_best_artifacts["model"].state_dict(), best_model_path)
        joblib.dump(stage2_best_artifacts["scaler"], best_scaler_path)

        y_true_sample = stage2_best_artifacts["Y_test"][0]
        y_pred_sample = stage2_best_artifacts["preds"][0]

        plot_prediction_vs_truth(
            y_true_sample,
            y_pred_sample,
            save_path=best_pred_plot_path,
            title="Final Best RNN: Prediction vs Ground Truth",
        )

        plot_losses(
            stage2_best_artifacts["train_losses"],
            stage2_best_artifacts["test_losses"],
            save_path=best_loss_plot_path,
            title="Final Best RNN: Loss Curve",
        )

        summary_txt = output_dir / f"best_rnn_two_stage_summary_{timestamp}.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("BEST RNN TWO-STAGE RESULT\n")
            f.write("=" * 40 + "\n")
            f.write(f"hidden_size: {stage2_best_result['hidden_size']}\n")
            f.write(f"num_layers: {stage2_best_result['num_layers']}\n")
            f.write(f"feature_set: {stage2_best_result['feature_set']}\n")
            f.write(f"exclude_columns: {stage2_best_result['exclude_columns']}\n")
            f.write(f"nonlinearity: {stage2_best_result['nonlinearity']}\n")
            f.write(f"num_epochs: {stage2_best_result['num_epochs']}\n")
            f.write(f"learning_rate: {stage2_best_result['learning_rate']}\n")
            f.write(f"dropout: {stage2_best_result['dropout']}\n")
            f.write(f"best_epoch: {stage2_best_result['best_epoch']}\n")
            f.write(f"best_test_loss: {stage2_best_result['best_test_loss']:.8f}\n")
            f.write(f"final_train_loss: {stage2_best_result['final_train_loss']:.8f}\n")
            f.write(f"final_test_loss: {stage2_best_result['final_test_loss']:.8f}\n")
            f.write(f"test_mse: {stage2_best_result['test_mse']:.8f}\n")
            f.write(f"test_mae: {stage2_best_result['test_mae']:.8f}\n")

        print("\nFinal best model summary:")
        print(stage2_best_result)
        print(f"\nSaved best model to: {best_model_path}")
        print(f"Saved best scaler to: {best_scaler_path}")
        print(f"Saved best prediction plot to: {best_pred_plot_path}")
        print(f"Saved best loss plot to: {best_loss_plot_path}")
        print(f"Saved summary text to: {summary_txt}")


if __name__ == "__main__":
    main()