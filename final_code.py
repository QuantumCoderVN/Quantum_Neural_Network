import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Đặt biến môi trường để khắc phục lỗi OpenMP (OMP) ---
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- Cấu hình thiết bị và Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64
N_SAMPLES = 2000
IMG_SIZE = 8

# --- Chuẩn bị thư mục lưu kết quả ---
RESULT_DIR = 'result'
if os.path.exists(RESULT_DIR):
    shutil.rmtree(RESULT_DIR)  # Xóa thư mục result nếu đã tồn tại
os.makedirs(RESULT_DIR)
print(f"Created directory: {RESULT_DIR}")

# --- Tải và xử lý dữ liệu MNIST ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_dataset.data = train_dataset.data[:N_SAMPLES]
train_dataset.targets = train_dataset.targets[:N_SAMPLES]

test_dataset.data = test_dataset.data[:N_SAMPLES // 5]
test_dataset.targets = test_dataset.targets[:N_SAMPLES // 5]

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Image dimensions: {IMG_SIZE}x{IMG_SIZE}")

# --- Định nghĩa Mạch Lượng tử (Quantum Circuit) ---
n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(pnp.pi * inputs[i], wires=i)

    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])

    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Định nghĩa Lớp Lượng tử (QuantumLayer) trong PyTorch ---
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super(QuantumLayer, self).__init__()
        self.qml_circuit = quantum_circuit
        n_layers_ansatz = 1
        self.weights = nn.Parameter(torch.rand(n_layers_ansatz, n_qubits, 3, requires_grad=True))

    def forward(self, x):
        batch_out = []
        for i in range(x.shape[0]):
            q_out_list = self.qml_circuit(x[i], self.weights)
            q_out_tensor = torch.stack(q_out_list)
            batch_out.append(q_out_tensor)
        return torch.stack(batch_out)

# --- Định nghĩa Mô hình QNN Lai (HybridQNN) ---
class HybridQNN(nn.Module):
    def __init__(self, img_size, n_qubits, num_classes):
        super(HybridQNN, self).__init__()
        self.img_size = img_size
        self.n_qubits = n_qubits

        self.classical_nn = nn.Sequential(
            nn.Linear(img_size * img_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits)
        )

        self.quantum_layer = QuantumLayer(n_qubits)

        self.post_quantum_nn = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classical_nn(x)
        x = self.quantum_layer(x)
        x = self.post_quantum_nn(x)
        return x

# Khởi tạo mô hình
num_classes = 10
model = HybridQNN(IMG_SIZE, n_qubits, num_classes).to(device)
print("\n--- Model Architecture ---")
print(model)

# --- Huấn luyện và Đánh giá Mô hình ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

num_epochs = 100

print("\n--- Training Started ---")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_accuracy = correct_test / total_test
    test_loss_list.append(test_loss)
    test_acc_list.append(test_accuracy)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

print("--- Training Complete ---")

# --- Lưu kết quả huấn luyện vào file text ---
with open(os.path.join(RESULT_DIR, 'training_results.txt'), 'w') as f:
    f.write("Epoch,Train Loss,Train Accuracy,Test Loss,Test Accuracy\n")
    for epoch in range(num_epochs):
        f.write(f"{epoch+1},{train_loss_list[epoch]:.4f},{train_acc_list[epoch]:.4f},{test_loss_list[epoch]:.4f},{test_acc_list[epoch]:.4f}\n")
print(f"Training results saved to {os.path.join(RESULT_DIR, 'training_results.txt')}")

# --- Hàm để hiển thị ảnh dự đoán ---
def plot_predictions(model, data_loader, num_images=10, filename="predictions.png"):
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))

    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    images_to_show = images[:num_images].to(device)
    labels_to_show = labels[:num_images].to(device)

    with torch.no_grad():
        outputs = model(images_to_show)
        _, predicted = torch.max(outputs.data, 1)

    for idx in range(num_images):
        ax = axes[idx]
        img = images_to_show[idx].cpu().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        true_label = labels_to_show[idx].item()
        pred_label = predicted[idx].item()
        is_correct = (true_label == pred_label)
        color = "green" if is_correct else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

# --- Gọi hàm hiển thị ảnh dự đoán ---
print("\n--- Displaying Predictions ---")
plot_predictions(model, test_loader, num_images=10, filename="test_predictions.png")

# --- Vẽ và Lưu biểu đồ Loss và Accuracy ---
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_list, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_loss_list, marker='x', linestyle='--', color='r', label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_list, marker='o', linestyle='-', color='g', label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_acc_list, marker='x', linestyle='--', color='m', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'training_metrics.png'))
plt.close()

# --- Bổ sung: Các thông số đánh giá chi tiết ---
print("\n--- Detailed Evaluation Metrics ---")

# Ma trận nhầm lẫn (Confusion Matrix)
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
plt.close()

# Báo cáo phân loại (Classification Report)
class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)], zero_division=0)
print("\nClassification Report:\n", class_report)

# Lưu báo cáo phân loại vào file
with open(os.path.join(RESULT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(class_report)
print(f"Classification report saved to {os.path.join(RESULT_DIR, 'classification_report.txt')}")