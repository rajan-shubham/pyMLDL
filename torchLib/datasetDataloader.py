
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