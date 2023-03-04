import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set here for your DIGIT
xxxx = "Dxxxx"
image_size = 224

all_image_dir = f'./D{xxxx}/'

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
}

# Specify the path to the directory where the data is located and transform.
all_dataset = torchvision.datasets.ImageFolder(root=all_image_dir, transform=data_transform['train'])

n_samples = len(all_dataset) # n_samples is 3000 (1500 for each class)
train_size = int(len(all_dataset) * 0.8) # train_size is 2400
val_size = n_samples - train_size # val_size is 600

# Shuffle the dataset and split it into train and validation sets.
train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])

# Create DataLoader for training and validation sets.
batch_size = 512
train_dataLoader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataLoader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders_dict = {
    'train': train_dataLoader, 
    'val': val_dataLoader
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp1 = nn.Linear(3*224*224, 500)
        self.mlp2 = nn.Linear(500, 128)
        self.mlp3 = nn.Linear(128, 2)

        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.reshape(-1, 3*224*224)
        x = self.bn1(F.relu(self.mlp1(x)))
        x = self.bn2(F.relu(self.mlp2(x)))
        x = self.mlp3(x)
        return x

net = Net()
print(net)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

num_epochs = 30
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()
        epoch_loss = 0.0
        epoch_corrects = 0
        # DataLoaderからデータをバッチごとに取り出す
        for inputs, labels in dataloaders_dict[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            # Initialize the optimizer
            optimizer.zero_grad()
            # Calculate the gradient only when training
            with torch.set_grad_enabled(phase == 'train'):
                 outputs = net(inputs)
                 loss = criterion(outputs, labels)
                 # Predict
                 soft = nn.Softmax(dim=1)
                 preds = torch.argmax(soft(outputs), dim=1)
                 # Backprop when training
                 if phase == 'train':
                     loss.backward()
                     # Update the parameters
                     optimizer.step()
                 epoch_loss += loss.item() * inputs.size(0)
                 epoch_corrects += torch.sum(preds == labels.data)

        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_accuracy = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))
        torch.save(net.state_dict(), f'weight/v2/D{xxxx}-epoch{epoch}.pth')
        
# Save full model
torch.save(net, f'weight/D20261-full.pth')