import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/EEG_Eye_State_Classification.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

features = np.reshape(features, (features.shape[0], 1, 1, features.shape[1]))
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=32, out_features=2, bias=True),
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classify(x)
        return x

def train():   
    model = EEGNet()

    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    batch = 32

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

    return model

if __name__ == '__main__':
    model = train()
    model.eval()

    with torch.no_grad():
        test_outputs = model(test_features)
    _, predicted_labels = torch.max(test_outputs.data, 1)
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
    print(f"Accuracy: {accuracy}")

    torch.save(model.state_dict(), 'eeg_model.pth')
