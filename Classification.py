"""
Pipeline:
1. Installing libraries
2. Loading the dataset
3. Checking the dimensions of training, validation, and testing
4. Checking label distribution
5. Zero-rule baseline (majority class classifier)
6. A quick visual check

7. Create a simple baseline using CNNs (2-3 layers)
8. Loss function, optimizer, metrics
9. Model training
10. Evaluate
12. Testing
"""
# 1. Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
from customDataset import FireAndNotFire
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# Hyperparameters
batch_size = 32

# 2. Creating a csv file from the fire_folder
fire_images_path = 'fire_dataset/fire_images'
non_fire_images_path = 'fire_dataset/non_fire_images'

# Collecting the data
data = []

# Collecting fire images and labeling them as 1
for filename in os.listdir(fire_images_path):
    if filename.endswith('.png'):
        data.append([os.path.join('fire_images', filename), 1])

# Collecting non-fire images and labeling them as 0
for filename in os.listdir(non_fire_images_path):
    if filename.endswith('.png'):
        data.append([os.path.join('non_fire_images', filename), 0])

# Create a dataframe from the given list
df = pd.DataFrame(data, columns=['image_name', 'label'])

# Saving our df to csv file
df.to_csv('fire_dataset/labels.csv', index=False)

# 3. Loading the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224 x 224
    transforms.ToTensor()  # Convert to tensor
])

dataset = FireAndNotFire(csv_file='fire_dataset/labels.csv', root_dir='fire_dataset',
                         transform=transform)

# Filter out invalid items
filtered_data = [item for item in dataset if item[0] is not None]

# Splitting the dataset
train_size = int(0.7 * len(filtered_data))
val_size = int(0.15 * len(filtered_data))
test_size = len(filtered_data) - train_size - val_size

train_set, val_set, test_set = random_split(filtered_data, [train_size, val_size, test_size])

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val_set,
                        batch_size=batch_size,
                        shuffle=False)

test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,
                         shuffle=False)

# 4. Check label distribution

from collections import Counter

train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())

print("\nTraining label distribution:")
print(sorted(train_counter.items()))

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())

print("\nValidation label distribution:")
print(sorted(val_counter.items()))

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\nTest label distribution:")
print(sorted(test_counter.items()))

"""
Our datasets clearly have an imbalanced datasets, so we can deal with it

using oversampling
"""

# 5. Zero-rule baseline (majority class classifier)
majority_class = test_counter.most_common(1)[0]
print("Majority class:", majority_class[0])

baseline_acc = majority_class[1] / sum(test_counter.values())
print("Accuracy when always predicting the majority class:")
print(f"{baseline_acc:.2f} ({baseline_acc * 100:.2f}%)")


# 6. A quick visual check
# for images, labels in train_loader:
#     break
#
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training images")
# plt.imshow(np.transpose(torchvision.utils.make_grid(
#     images[:64],
#     padding=2,
#     normalize=True),
#     (1, 2, 0)))
# plt.show()


# 7. Training a baseline model
class CNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Calculate the size of the feature maps after convolutions
        self.feature_size = self._get_conv_output_size(input_channels, 224, 224)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _get_conv_output_size(self, in_channels, height, width):
        x = torch.randn(1, in_channels, height, width)
        x = self.conv(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


baseline_model = CNN(input_channels=3).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

epochs = 10
train_loss = []
val_loss = []
train_acc = []
val_acc = []
best_loss = None

for epoch in range(epochs):
    baseline_model.train()
    running_train_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    train_loop = tqdm(train_loader, leave=False)

    for x, targets in train_loop:
        x = x.to(device)
        targets = targets.unsqueeze(1).float().to(device)  # Ensure targets shape for BCELoss

        optimizer.zero_grad()
        predictions = baseline_model(x)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * x.size(0)  # Track total loss across batches
        predicted_labels = (predictions >= 0.5).squeeze().long()  # Threshold predictions
        correct_predictions += (
                predicted_labels == targets.squeeze().long()).sum().item()  # Calculate correct predictions
        total_samples += targets.size(0)  # Track total number of samples

        train_loop.set_description(
            f"Epoch [{epoch + 1}/{epochs}], train_loss={running_train_loss / total_samples:.4f}, train_acc={correct_predictions / total_samples:.4f}")

    mean_training_loss = running_train_loss / len(train_loader.dataset)
    train_loss.append(mean_training_loss)
    train_acc.append(correct_predictions / total_samples)

    baseline_model.eval()
    running_val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for x, targets in val_loader:
            x = x.to(device)
            targets = targets.unsqueeze(1).float().to(device)

            predictions = baseline_model(x)
            loss = criterion(predictions, targets)
            running_val_loss += loss.item() * x.size(0)

            predicted_labels = (predictions >= 0.5).squeeze().long()
            correct_predictions += (predicted_labels == targets.squeeze().long()).sum().item()
            total_samples += targets.size(0)

        mean_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss.append(mean_val_loss)
        val_acc.append(correct_predictions / total_samples)

        print(f"Epoch [{epoch + 1}/{epochs}], train_loss={mean_training_loss:.4f}, train_acc={train_acc[-1]:.4f}, "
              f"val_loss={mean_val_loss:.4f}, val_acc={val_acc[-1]:.4f}")

        if best_loss is None or mean_val_loss < best_loss:
            best_loss = mean_val_loss

            checkpoint = {
                'state_model': baseline_model.state_dict(),
                'state_optimizer': optimizer.state_dict(),
                'state_lr_scheduler': lr_scheduler.state_dict(),
                'loss': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss
                },
                'metric': {
                    'train_acc': train_acc,
                    'val_acc': val_acc
                },
                'epochs': {
                    'epochs': epochs,
                    'save_epoch': epoch
                }
            }

            torch.save(checkpoint, f'model_state_dict_epoch_{epoch + 1}.pt')
            print(f'At epoch - {epoch + 1}, model was saved with validation loss of - {mean_val_loss:.4f}', end='\n\n')

# Plotting the loss graphs for training and validation
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['loss_train', 'loss_val'])
plt.show()

# Plotting the acc graphs for training and validation
plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['acc_train', 'acc_val'])
plt.show()

baseline_model.eval()

# Define lists to accumulate predictions and ground truth labels
all_predictions = []
all_targets = []

# Iterate over test data
with torch.no_grad():
    for x, targets in test_loader:
        x = x.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = baseline_model(x)

        # Threshold predictions (assuming binary classification with sigmoid)
        binary_predictions = (predictions >= 0.5).squeeze().cpu().numpy().astype(int)

        # Accumulate predictions and ground truth labels
        all_predictions.extend(binary_predictions)
        all_targets.extend(targets.cpu().numpy())

# Convert lists to numpy arrays for easier computation
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Calculate accuracy
accuracy = np.mean((all_predictions == all_targets))

# Print the overall accuracy
print(f"Overall accuracy on test set: {accuracy:.4f}")

# You can compute more metrics like precision, recall, F1-score, etc., using sklearn.metrics

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(all_targets, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Set model to evaluation mode
baseline_model.eval()

# Initialize lists to store predictions and true labels
pred = []
true_labels = []

# Iterate through the test set and collect predictions and true labels
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = baseline_model(images)
        predictions = (outputs > 0.5).float().cpu().numpy().flatten()  # Assuming a threshold of 0.5 for binary classification
        pred.extend(predictions)
        true_labels.extend(labels.cpu().numpy())

# Convert true_labels to numpy array for comparison
true_labels = np.array(true_labels)

# Display 15 random images with their true and predicted labels
random_indices = np.random.randint(0, len(test_set), 15)

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    image, true_label = test_set[random_indices[i]]
    image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
    ax.imshow(image)
    predicted_label = int(pred[random_indices[i]])
    if true_label == predicted_label:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {true_label}\nPredicted: {predicted_label}", color=color)

plt.tight_layout()
plt.show()

pred = np.array(pred)

# Find indices where predictions are incorrect
incorrect_indices = np.where(pred != true_labels)[0]

# Print incorrect predictions
print("Incorrect predictions:")
for i in incorrect_indices:
    image, true_label = test_set[i]
    image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
    predicted_label = int(pred[i])
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")

    plt.imshow(image)
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.show()
