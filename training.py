import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # For data loading and batching
import torch.nn.functional as F

training = True


def get_accuracy(pred, test):
    correct = 0
    wrong = 0
    for p, t in zip(torch.argmax(pred,1), test):
        if p == t:
            correct+=1
        else:
            wrong+=1
    return (correct/test.shape[0])*100


class LSTMMultiClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob=0.5):
        super(LSTMMultiClass, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=dropout_prob, num_layers=n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # out = self.dropout(lstm_out[:, -1, :])
        out = self.fc1(lstm_out[:, -1, :])
        out = self.dropout(out)
        out = self.fc2(out)
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
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, integer_labels, test_size=0.2, random_state=42)

print("\nSplitted dataset and labels: ")
print(f'\t{train_dataset.shape=}')
print(f'\t{test_dataset.shape=}')
print(f'\t{train_labels.shape=}')
print(f'\t{test_labels.shape=}')

print("\n--- Training ---")
# Set device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nSetting torch device to: {device=}')

x = torch.Tensor(train_dataset).to(device)
y = torch.Tensor(train_labels).squeeze().long().to(device)

x_test = torch.Tensor(test_dataset).to(device)
y_test = torch.Tensor(test_labels).squeeze().long().to(device)
print(f'\t{x.shape=}')
print(f'\t{y.shape=}')

# Define hyperparameters
input_dim = 24
hidden_dim = 128
n_layers = 1
output_dim = y.unique().shape[0]
lr = 0.001
epochs = 200
batch_size = 16
dropout = 0.2
l2_lambda = 0.00005

print(f'\nHyperparameters: ')
print(f'\t{input_dim=}')
print(f'\t{hidden_dim=}')
print(f'\t{output_dim=}')
print(f'\t{lr=}')
print(f'\t{epochs=}')
print(f'\t{batch_size=}\n')

# Instantiate the model
model = LSTMMultiClass(input_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
print(f'{model=}')

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

# Create a DataLoader for the input data and labels
print(f'{x.shape=}')
print(f'{y.shape=}')
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print(f'{x_train.shape=}')
print(f'{y_train.shape=}')
print(f'{x_val.shape=}')
print(f'{y_val.shape=}')
data_train = TensorDataset(x_train, y_train)
data_val = TensorDataset(x_val, y_val)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)


# Set up early stopping
patience = 10
best_val_loss = float('inf')
counter = 0

if training:
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
        acc = get_accuracy(y_pred, y_val)
        print(f"Epoch {epoch}: Training Loss = {loss.item():.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {acc:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "lstm_model.pth")
            counter = 0
        else:
            counter += 1
        if counter == patience:
            print(f"Stopping training after {epoch} epochs due to no improvement in validation loss.")
            break


model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()
y_pred = model(x_test)  

acc = get_accuracy(y_pred, y_test)
print("accuracy: ", acc)
