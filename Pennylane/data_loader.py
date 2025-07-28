# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, N_TRAIN_SAMPLES, N_TEST_SAMPLES, IMG_SIZE

def get_mnist_data_loaders():
    """
    Tải và tiền xử lý dữ liệu MNIST, sau đó tạo DataLoader.

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Giới hạn số lượng mẫu để huấn luyện nhanh hơn cho mục đích minh họa
    train_dataset.data = train_dataset.data[:N_TRAIN_SAMPLES]
    train_dataset.targets = train_dataset.targets[:N_TRAIN_SAMPLES]

    test_dataset.data = test_dataset.data[:N_TEST_SAMPLES]
    test_dataset.targets = test_dataset.targets[:N_TEST_SAMPLES]

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Image dimensions: {IMG_SIZE}x{IMG_SIZE}")

    return train_loader, test_loader