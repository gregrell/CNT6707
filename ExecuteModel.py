from RellPytorch.FeatureDataset import RobotArmDataset
from torch.utils.data.dataloader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# This code used to check the NN model to make sure it is vaguely doing what it is supposed to do
model = NN(4,2)
x = torch.randn(64,4)
print(model(x).shape)

# set device
device = torch.device('cpu')

# hyperparameters
input_size = 4
num_classes = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 10


# load in custom dataset from a custom dataset object in RellPytorch module
feature_data = RobotArmDataset('data/TrainingData.csv')
print(feature_data.__len__())

Train_Data = DataLoader(feature_data, batch_size=batch_size, shuffle=True)
Test_Data = DataLoader(feature_data, batch_size=batch_size, shuffle=True)

'''
for features, label in Train_Data:
    # print(f"features{features}")
    print(f"label{label}")
'''
# initialize the network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

check_accuracy(Train_Data, model)



