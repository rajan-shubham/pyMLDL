import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# set random seed for reproducibility
torch.manual_seed(42)

# read the csv file
df = pd.read_csv('fmnist_small.csv') # using right now 6k images from full dataset
print(df.head())

# train test split
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# scaling the features
X_train = X_train/255.0
X_test = X_test/255.0

# create CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# create train_dataset object
train_dataset = CustomDataset(X_train, y_train)

# create test_dataset object
test_dataset = CustomDataset(X_test, y_test)

# create dataloader object
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# define NN class
class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            # no need to add softmax as we are using CrossEntropyLoss
        )
    
    def forward(self, x):
        return self.model(x)
    
# set learning rate and epochs
epochs = 100
learning_rate = 0.1

# instantiate model
model = MyNN(X_train.shape[1])

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(batch_features)
        # calculate loss
        loss = criterion(outputs, batch_labels)
        # backward pass
        loss.backward()
        # update weights/grads
        optimizer.step()

        # calculate total epoch loss
        total_epoch_loss += loss.item()
    
    # print epoch loss
    # har epochs me per batch me jitna loss aaya usko average karke print
    print(f"Epoch {epoch+1}, Loss: {total_epoch_loss/len(train_loader)}")



# set model to evaluation mode
model.eval()

# initialize variables to keep track of correct predictions and total predictions
correct = 0
total = 0

# disable gradient calculation
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        # forward pass
        outputs = model(batch_features)
        # get predicted class
        _, predicted = torch.max(outputs.data, 1)
        # update correct and total
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

# print accuracy
print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")


"""
to maximize the acuaracy
1. use full dataset
2. use more epochs
3. use more hidden layers
4. use more neurons in hidden layers
5. use different optimizer
6. use different learning rate
7. use different batch size
8. use different activation function
9. use different loss function
10. use different weight initialization techniques

change in model architecture by 
1. adding dropout layer
2. adding batch normalization layer
3. adding residual connections
4. adding skip connections
5. adding attention mechanism
6. hyperparameter tuning
"""