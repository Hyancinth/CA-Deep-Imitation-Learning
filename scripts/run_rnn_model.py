import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
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


def plot_losses(train_losses, test_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("RNN Training and Test Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(y_true, y_pred, save_path=None, title_suffix=""):
    t = np.arange(y_true.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(t, y_true[:, 0], label="True u1")
    plt.plot(t, y_pred[:, 0], label="Pred u1", linestyle="--")
    plt.plot(t, y_true[:, 1], label="True u2")
    plt.plot(t, y_pred[:, 1], label="Pred u2", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Control Value")
    plt.title(f"RNN Predictions vs Ground Truth {title_suffix}")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def evaluate_on_test_set(model, x_test, y_test, device):
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()

    mse = np.mean((preds - y_test) ** 2)
    mae = np.mean(np.abs(preds - y_test))

    return preds, mse, mae


def main():
    training_file_path = "C:\\Bita Drive\\1. McMaster\\Deep Learning\\Final Project\\CA-Deep-Imitation-Learning\\model\\data\\data_320_01_100.h5"
    output_dir = "model/rnn_outputs"
    model_dir = "model/trained_models"
    scaler_dir = "model/scalers"

    ensure_dir(output_dir)
    ensure_dir(model_dir)
    ensure_dir(scaler_dir)

    # -----------------------------
    # hyperparameters
    # -----------------------------
    exclude_columns = []
    # examples:
    # exclude_columns = ["u1_prev", "u2_prev"]
    # exclude_columns = ["u1_prev", "u2_prev", "ee_dx_target", "ee_dy_target"]

    hidden_size = 64
    num_layers = 2
    dropout = 0.1
    nonlinearity = "tanh"   # "tanh" or "relu"
    bidirectional = False
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-5
    test_size = 0.2
    random_state = 42

    data = load_h5_runs(training_file_path)

    X, Y, used_x_columns, y_columns = build_sequences_from_runs(
        data,
        exclude_columns=exclude_columns,
    )

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Used input columns:", used_x_columns)

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
        bidirectional=bidirectional,
    )

    model, train_losses, test_losses, device = train_sequence_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    preds, mse, mae = evaluate_on_test_set(model, X_test_scaled, Y_test, device)

    print(f"\nFinal Test MSE: {mse:.6f}")
    print(f"Final Test MAE: {mae:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(model_dir, f"rnn_model_{timestamp}.pt")
    scaler_path = os.path.join(scaler_dir, f"rnn_scaler_{timestamp}.pkl")
    metrics_path = os.path.join(output_dir, f"rnn_metrics_{timestamp}.txt")
    loss_plot_path = os.path.join(output_dir, f"rnn_loss_plot_{timestamp}.png")
    pred_plot_path = os.path.join(output_dir, f"rnn_prediction_plot_{timestamp}.png")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training file: {training_file_path}\n")
        f.write(f"Input columns: {used_x_columns}\n")
        f.write(f"Output columns: {y_columns}\n")
        f.write(f"hidden_size={hidden_size}\n")
        f.write(f"num_layers={num_layers}\n")
        f.write(f"dropout={dropout}\n")
        f.write(f"nonlinearity={nonlinearity}\n")
        f.write(f"bidirectional={bidirectional}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"num_epochs={num_epochs}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"weight_decay={weight_decay}\n")
        f.write(f"test_size={test_size}\n")
        f.write(f"random_state={random_state}\n")
        f.write(f"Final Test MSE={mse:.6f}\n")
        f.write(f"Final Test MAE={mae:.6f}\n")

    plot_losses(train_losses, test_losses, save_path=loss_plot_path)

    y_true_sample = Y_test[0]
    y_pred_sample = preds[0]
    plot_sample_predictions(
        y_true_sample,
        y_pred_sample,
        save_path=pred_plot_path,
        title_suffix="(First Test Run)"
    )

    print("\nSaved files:")
    print("Model:", model_path)
    print("Scaler:", scaler_path)
    print("Metrics:", metrics_path)
    print("Loss plot:", loss_plot_path)
    print("Prediction plot:", pred_plot_path)


if __name__ == "__main__":
    main()