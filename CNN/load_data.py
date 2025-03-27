import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot

batch_size = 128

def load_data():
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    train_dataset = datasets.MNIST(root='./CNN/data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./CNN/data', train=False, transform=transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Jumlah data train: {len(train_dataset)}")
    print(f"Jumlah data test: {len(test_dataset)}")
    
    return train_loader, test_loader

def display_data_samples(data_loader, num_samples=3):
    class_samples = {i: [] for i in range(10)}
    
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            if len(class_samples[label.item()]) < num_samples:
                class_samples[label.item()].append(img)
        if all(len(class_samples[i]) >= num_samples for i in range(10)):
            break
    plt.figure(figsize=(15, 10))
    for cls in range(10):
        for i in range(num_samples):
            plt.subplot(10, num_samples, i + 1 + cls * num_samples)
            plt.imshow(class_samples[cls][i].squeeze().numpy(), cmap='gray')
            plt.title(f"Label: {cls}")
            plt.axis('off')
    plt.show()

