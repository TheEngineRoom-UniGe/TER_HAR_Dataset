import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # For data loading and batching
import torch.nn.functional as F

class LSTMMultiClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5, l2_lambda=0.01):
        super(LSTMMultiClass, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)

        # L2 regularization
        l2_reg = torch.tensor(0.)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)

        out += self.l2_lambda * l2_reg
        return out


print("\n--- Data Loading ---")

dataset = np.load('data_shape(1264_5628_24).npy')
labels = np.load('labels_shape(1264_1).npy')

# Convert string labels to integer labels
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels.ravel())

print("Loaded dataset and labels: ")
print(f'\t{dataset.shape=}')
print(f'\t{integer_labels.shape=}')

# Split the data into training and test sets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)

print("\nSplitted dataset and labels: ")
print(f'\t{train_dataset.shape=}')
print(f'\t{test_dataset.shape=}')
print(f'\t{train_labels.shape=}')
print(f'\t{test_labels.shape=}')

print("\n--- Training ---")
# Set device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nSetting torch device to: {device=}')

x = torch.Tensor(dataset).to(device)
y = torch.Tensor(integer_labels).squeeze().long().to(device)
print(f'\t{x.shape=}')
print(f'\t{y.shape=}')

# Define hyperparameters
input_dim = 24
hidden_dim = 64
output_dim = y.unique().shape[0]
lr = 0.005
epochs = 200
batch_size = 8
dropout = 0.5
l2_lambda = 0.01

print(f'\nHyperparameters: ')
print(f'\t{input_dim=}')
print(f'\t{hidden_dim=}')
print(f'\t{output_dim=}')
print(f'\t{lr=}')
print(f'\t{epochs=}')
print(f'\t{batch_size=}\n')

# Instantiate the model
model = LSTMMultiClass(input_dim, hidden_dim, output_dim, dropout, l2_lambda).to(device)
print(f'{model=}')

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create a DataLoader for the input data and labels
x_train, y_train, x_val, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
data_train = TensorDataset(x_train, y_train)
data_val = TensorDataset(x_val, y_val)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)


# Set up early stopping
patience = 5
best_val_loss = float('inf')
counter = 0

for epoch in range(epochs):
    # At the beginning of each epoch, set your model to train mode
    model.train()
    
    # Loop through your training data
    for i, batch in enumerate(train_loader):
        # Unpack the batch
        batch_x, batch_y = batch

        # Reset the gradients to zero
        optimizer.zero_grad()
        
        # Pass the input sequence through the model and get the predicted output
        output = model(batch_x)
        
        # Calculate the loss between the predicted output and the true output
        loss = criterion(output, batch_y)
        
        # Backpropagate the loss through the network and update the model parameters
        loss.backward()
        optimizer.step()

    model.eval()

    y_pred = model(x_val)
    val_loss = criterion(y_pred, y_val)
    print(f"Epoch {epoch}: Training Loss = {loss.item():.4f}, Validation Loss = {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(best_model.state_dict(), "lstm_model.pth")
        counter = 0
    else:
        counter += 1
    if counter == patience:
        print(f"Stopping training after {epoch} epochs due to no improvement in validation loss.")
        break

    