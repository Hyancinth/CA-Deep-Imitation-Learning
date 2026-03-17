import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def create_data_loaders(x_train, y_train, x_test, y_test, batch_size=32):
    
    # convert to tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # convert into PyTorch datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model: nn.Module, train_loader, test_loader, num_epochs=200, learning_rate=0.001):
    # define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # use small learning rate for adam

    train_losses = torch.zeros(num_epochs)
    test_losses = torch.zeros(num_epochs)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            # forward pass and loss
            pred = model(x_batch)
            loss = loss_function(pred, y_batch)

            # backpropagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses[epoch] = train_loss

        # evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                loss = loss_function(outputs, y_batch)
                test_loss += loss.item() * x_batch.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses[epoch] = test_loss

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return model, train_losses, test_losses