import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Step 1: Preparing the CSV Data
data = pd.read_csv('data/EEG_Eye_State_Classification.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Step 2: Data Preprocessing
features = np.reshape(features, (features.shape[0], 1, features.shape[1]))
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(16 * ((features.shape[-1] - 2)//2), 64)
        self.l2 = nn.Linear(64, 2)

    def forward(self, x):
        x= self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    
if __name__ == '__main__':
    model = CNN()

    # Step 5: Model Compilation and Training
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    batch = 50

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_features), batch):
            inputs = train_features[i:i+batch]
            targets = train_labels[i:i+batch]
            outputs = model(inputs)
            loss = lossfunc(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    model.eval()

    # Step 6: Model Evaluation
    with torch.no_grad():
        test_outputs = model(test_features)
    _, predicted_labels = torch.max(test_outputs.data, 1)
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
    print(f"Accuracy: {accuracy}")

    # Step 7: Model Deployment and Prediction
    torch.save(model.state_dict(), 'eeg_model.pth')
