import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20
batch_size = 25
learning_rate = 0.0001

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FruitsAndVegetablesRecognizer(nn.Module):

    def __init__(self, num_classes, init_method="xavier"):
        super(FruitsAndVegetablesRecognizer, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding=1)  # Output: 256x256x16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=1)  # Output: 256x256x32
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, padding=1)  # Output: 256x256x64
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2, padding=1)  # Output: 256x256x64
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 96, kernel_size=2, padding=1)  # Output: 256x256x96
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 128, kernel_size=2, padding=1)  # Output: 256x256x128
        self.bn6 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces dimensions by half

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, num_classes)

        self.initialize_weights(init_method)

    def forward(self, x):

        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x))))
        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x))))
        x = self.pool(
            F.relu(
                self.bn3(
                    self.conv3(x))))
        x = self.pool(
            F.relu(
                self.bn4(
                    self.conv4(x))))
        x = self.pool(
            F.relu(
                self.bn5(
                    self.conv5(x))))
        x = self.pool(
            F.relu(
                self.bn6(
                    self.conv6(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def initialize_weights(self, init_method):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_method == "xavier":
                    init.xavier_uniform_(m.weight)
                elif init_method == "small_random":
                    init.uniform_(m.weight, a=-0.01, b=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.1)

    def calculate_metrics(self, data_loader, criterion, device):
        total_loss = 0.0
        all_preds = []
        all_labels = []

        self.eval()

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = self(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        average_loss = total_loss / len(data_loader)

        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f'Average Loss: {average_loss:.6f}')
        print(f'Precision (Macro): {precision:.4f}')
        print(f'Recall (Macro): {recall:.4f}')
        print(f'F1-Score (Macro): {f1:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')

        return {
            'loss': average_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_matrix': conf_matrix
        }


model = FruitsAndVegetablesRecognizer(
    num_classes=36,
    init_method="small_random",
).to(device)

summary(model, input_size=(3, 256, 256), batch_size=20)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

train_accuracies = []
val_accuracies = []
test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0

    # Training
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%')

# Validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            _, predicted = torch.max(val_outputs.data, 1)
            correct_val += (predicted == val_labels).sum().item()
            total_val += val_labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')
    scheduler.step(val_accuracy)


# Test
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_outputs = model(test_images)
        _, predicted = torch.max(test_outputs.data, 1)
        correct_test += (predicted == test_labels).sum().item()
        total_test += test_labels.size(0)

test_accuracy = 100 * correct_test / total_test
test_accuracies.append(test_accuracy)
print(f'Final Test Accuracy: {test_accuracy:.2f}%')

test_metrics = model.calculate_metrics(test_loader, criterion, device)

base_results_dir = "results"
if not os.path.exists(base_results_dir):
    os.makedirs(base_results_dir)

existing_folders = [int(d) for d in os.listdir(base_results_dir) if d.isdigit()]
next_folder_num = max(existing_folders, default=0) + 1
current_results_dir = os.path.join(base_results_dir, str(next_folder_num))
os.makedirs(current_results_dir)

metrics_file = os.path.join(current_results_dir, "metrics.txt")
with open(metrics_file, 'w') as f:
    f.write("Epoch, Train Accuracy, Val Accuracy, Test Accuracy\n")
    for epoch in range(num_epochs):
        f.write(f"{epoch + 1}, {train_accuracies[epoch]:.2f}, {val_accuracies[epoch]:.2f}, {test_accuracy:.2f}\n")

    f.write("\nFinal Test Metrics:\n")
    f.write(f"Test Loss: {test_metrics['loss']:.6f}\n")
    f.write(f"Precision (Macro): {test_metrics['precision']:.4f}\n")
    f.write(f"Recall (Macro): {test_metrics['recall']:.4f}\n")
    f.write(f"F1-Score (Macro): {test_metrics['f1']:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(test_metrics['conf_matrix'], separator=', '))
print(f"Metrics saved to {metrics_file}")

output_file = os.path.join(current_results_dir, "accuracy_plot.png")
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='g', linestyle='-', label='Test Accuracy')
plt.title('Training, Validation, and Test Losses')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig(output_file)
print(f'Accuracy plot saved as: {output_file}')
plt.show()
