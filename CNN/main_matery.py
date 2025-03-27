import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from torchviz import make_dot
import os
import matplotlib.pyplot as plt
from LeNet import LeNet
from load_data import load_data

batch_size = 128
epochs = 10
learning_rate = 0.01
momentum = 0.9
modelPath = "./CNN/cnn_lenet_trained.pth"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    print(f"Training model on {len(train_loader)}")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')
    save_model(model, modelPath)


def load_trained_model(path):
    model = LeNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def visualize_prediction(images, labels, predictions):
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Pred: {predictions[i]}, Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    images_batch, labels_batch, predictions = None, None, None
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if images_batch is None:
                images_batch = images
                labels_batch = labels
                predictions = predicted
            break
    accuracy = 100 * correct / total
    print(f"Accuracy of the model: {accuracy}%")
    visualize_prediction(images_batch, labels_batch, predictions)


def main():
    train_loader, test_loader = load_data()
    if os.path.exists(modelPath):
        print("Model found, loading model...")
        model = load_trained_model(modelPath)
        evaluate_model(model, test_loader, device)
    else:
        print("Model not found, training model...")
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        train_model(model, train_loader, criterion, optimizer, device)
        evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
