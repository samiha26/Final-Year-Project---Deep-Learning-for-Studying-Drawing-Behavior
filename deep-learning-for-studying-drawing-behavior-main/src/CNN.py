import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define your CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output before the fully connected layer
        x = self.fc1(x)
        return x

# Set the device to use as the GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the number of classes based on your problem
num_classes = 3

# Create an instance of the CNN model
cnn_model = SimpleCNN(num_classes=num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Assuming you have a custom dataset class named DoodleDatasetSimple
# Define your data transforms, e.g., resizing and normalization
your_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Specify the path to your CNN data
your_cnn_data_path = '../images/house/'

# Create an instance of your DoodleDatasetSimple
class DoodleDatasetSimple(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for file_name in os.listdir(self.root_dir):
            img_path = os.path.join(self.root_dir, file_name)
            samples.append(img_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Assuming the file name format is "<label>.png"
        label = int(img_path.split('/')[-1].split('.')[0])

        return {'image': img, 'class': label}

# Example usage
your_data_path = '../images/house/'
your_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

custom_dataset = DoodleDatasetSimple(root_dir=your_data_path, transform=your_transform)

# Split your dataset into training and validation sets
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

# Create DataLoader instances for training and validation
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    cnn_model.train()
    for batch in train_loader:
        images, labels = batch['image'].to(device), batch['class'].to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch['image'].to(device), batch['class'].to(device)

            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy * 100:.2f}%')