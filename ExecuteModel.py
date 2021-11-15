from RellPytorch.FeatureDataset import RobotArmDataset
from torch.utils.data.dataloader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path

ModelPath = 'data\modelData.dat'


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# set device
device = torch.device('cpu')

# hyperparameters
input_size = 4
num_classes = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 1000


# initialize the network
model = NN(input_size=input_size, num_classes=num_classes).to(device)


# if a trained model file does not exist then train the model
modelExists = Path(ModelPath).is_file()
if not modelExists:
    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # load in custom dataset from a custom dataset object in RellPytorch module
    feature_data = RobotArmDataset('data/TrainingData.csv')
    print(f'Imported robot arm dataset with {feature_data.__len__()} data points')
    Train_Data = DataLoader(feature_data, batch_size=batch_size, shuffle=True)
    Test_Data = DataLoader(feature_data, batch_size=batch_size, shuffle=True)
    # Train the network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(Train_Data):
            data = data.to(device=device)
            targets = targets.to(device=device)


            # Forward
            scores = model(data)
            # Change the target data type from float to long because that's what cross entropy loss needs
            targets = targets.long()
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
else:
    model.load_state_dict(torch.load(ModelPath))
    print(f'Trained model data loaded from {Path(ModelPath)}')


# Check the accuracy on training, check to see how good the model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()


if not modelExists:
    check_accuracy(Train_Data, model)



# the following data should predict a 1 by the model


# 56.843,70.096,42.562,176.912


if not Path(ModelPath).is_file():
    # Save the model
    torch.save(model.state_dict(), ModelPath)
    print(f'Model Saved at {Path(ModelPath)}')


