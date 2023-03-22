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
data = TensorDataset(x, y)
loader = DataLoader(data, batch_size=batch_size, shuffle=True)
validation_set = ...

# Train the model
for epoch in range(epochs):
    running_loss = 0.0
    for i, batch in enumerate(loader):
        # Unpack the batch
        batch_x, batch_y = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(batch_x)
        loss = criterion(output, batch_y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    # Print the loss every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "lstm_model.pth")
    accuracy = 100 * correct / total
    print(f'Epoch {epoch}/{epochs} - Loss: {running_loss / len(loader):.4f} - Accuracy: {accuracy:.2f}%')
