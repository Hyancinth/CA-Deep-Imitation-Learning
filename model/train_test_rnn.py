import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_sequence_data_loaders(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=16,
):
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_sequence_model(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = loss_function(preds, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(data_loader.dataset)


def train_sequence_model(
    model,
    train_loader,
    test_loader,
    num_epochs=200,
    learning_rate=1e-3,
    weight_decay=0.0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = loss_function(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_test_loss = evaluate_sequence_model(model, test_loader, loss_function, device)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Test Loss: {avg_test_loss:.6f}"
        )

    return model, train_losses, test_losses, device