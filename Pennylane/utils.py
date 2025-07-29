# utils.py

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch 
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pennylane as qml
from config import (
    RESULT_DIR, LOG_FILE, QUANTUM_CIRCUIT_PLOT_PATH,
    TRAINING_METRICS_PLOT_PATH, QUANTUM_WEIGHTS_PLOT_PATH,
    PREDICTIONS_PLOT_PATH, CONFUSION_MATRIX_PLOT_PATH,
    CLASSIFICATION_REPORT_FILE, N_QUBITS, NUM_CLASSES, N_LAYERS_ANSATZ
)
from pennylane import numpy as pnp # Dùng pnp cho các tác vụ vẽ mạch Pennylane

def setup_results_directory():
    """Tạo hoặc làm sạch thư mục lưu kết quả."""
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)
    print(f"Created directory: {RESULT_DIR}")

def write_log(message, filename=LOG_FILE):
    """Ghi thông điệp vào file log."""
    filepath = os.path.join(RESULT_DIR, filename)
    with open(filepath, 'a') as f:
        f.write(message + '\n')

def plot_quantum_circuit(qnode_func, filename=QUANTUM_CIRCUIT_PLOT_PATH):
    """
    Vẽ và lưu kiến trúc mạch lượng tử.
    Args:
        qnode_func (qml.qnode): Hàm QNode để vẽ.
        filename (str): Tên file để lưu biểu đồ.
    """
    print("\n--- Plotting Quantum Circuit ---")
    # Cung cấp các giá trị đầu vào và trọng số giả cho QNode để vẽ
    dummy_inputs = pnp.random.rand(N_QUBITS, requires_grad=False)
    dummy_weights = pnp.random.rand(N_LAYERS_ANSATZ, N_QUBITS, 3, requires_grad=False)

    fig, ax = qml.draw_mpl(qnode_func)(dummy_inputs, dummy_weights)
    ax.set_title("Quantum Circuit Architecture")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Quantum circuit plot saved to {filename}")

def plot_predictions(model, data_loader, device, num_images=10, filename=PREDICTIONS_PLOT_PATH):
    """
    Hiển thị và lưu các dự đoán của mô hình trên một số ảnh mẫu.
    Args:
        model (torch.nn.Module): Mô hình đã huấn luyện.
        data_loader (torch.utils.data.DataLoader): DataLoader chứa dữ liệu test.
        device (torch.device): Thiết bị (CPU/GPU) để chạy inference.
        num_images (int): Số lượng ảnh muốn hiển thị.
        filename (str): Tên file để lưu biểu đồ.
    """
    print("\n--- Displaying Predictions ---")
    model.eval()
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
    plt.savefig(filename)
    plt.close()
    print(f"Predictions plot saved to {filename}")

def plot_training_metrics(train_loss, train_acc, test_loss, test_acc, num_epochs, filename=TRAINING_METRICS_PLOT_PATH):
    """
    Vẽ và lưu biểu đồ Loss và Accuracy trong quá trình huấn luyện.
    Args:
        train_loss (list): Danh sách loss trên tập huấn luyện qua các epoch.
        train_acc (list): Danh sách accuracy trên tập huấn luyện qua các epoch.
        test_loss (list): Danh sách loss trên tập test qua các epoch.
        test_acc (list): Danh sách accuracy trên tập test qua các epoch.
        num_epochs (int): Tổng số epoch.
        filename (str): Tên file để lưu biểu đồ.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss, marker='o', linestyle='--', color='r', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc, marker='o', linestyle='-', color='g', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_acc, marker='o', linestyle='--', color='m', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Training metrics plot saved to {filename}")

def plot_quantum_weights_evolution(quantum_weights_history, num_epochs, filename=QUANTUM_WEIGHTS_PLOT_PATH):
    """
    Vẽ biểu đồ sự thay đổi của các tham số lượng tử qua các epoch.
    Args:
        quantum_weights_history (list): Danh sách các giá trị trọng số lượng tử tại mỗi epoch.
        num_epochs (int): Tổng số epoch.
        filename (str): Tên file để lưu biểu đồ.
    """
    plt.figure(figsize=(14, 8))
    plt.title('Quantum Layer Weights Evolution over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.grid(True)

    if not quantum_weights_history:
        print("No quantum weights history to plot.")
        return

    quantum_weights_array = np.array(quantum_weights_history)
    # quantum_weights_array có shape (num_epochs, n_layers_ansatz, n_qubits, 3)
    num_total_quantum_params = quantum_weights_array.reshape(num_epochs, -1).shape[1]
    reshaped_weights = quantum_weights_array.reshape(num_epochs, -1)

    for i in range(num_total_quantum_params):
        plt.plot(range(1, num_epochs + 1), reshaped_weights[:, i], alpha=0.7) # Bỏ label để tránh quá tải legend

    if num_total_quantum_params <= 12:
        # Nếu số lượng tham số ít, thêm legend để xác định từng đường
        labels = []
        for l_idx in range(N_LAYERS_ANSATZ):
            for q_idx in range(N_QUBITS):
                for param_idx in range(3):
                    labels.append(f'L{l_idx}Q{q_idx}P{param_idx}')
        plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.text(num_epochs * 0.5, plt.ylim()[1] * 0.9,
                 f'Showing {num_total_quantum_params} quantum weights',
                 horizontalalignment='center', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Quantum weights evolution plot saved to {filename}")


def evaluate_and_report(all_labels, all_preds):
    """
    Tính toán và lưu ma trận nhầm lẫn và báo cáo phân loại.
    Args:
        all_labels (list): Danh sách các nhãn thực tế.
        all_preds (list): Danh sách các nhãn dự đoán.
    """
    print("\n--- Detailed Evaluation Metrics ---")

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PLOT_PATH}")

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)], zero_division=0)
    print("\nClassification Report:\n", class_report)

    with open(CLASSIFICATION_REPORT_FILE, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to {CLASSIFICATION_REPORT_FILE}")


# HybridQNN Parameter Counting
def count_parameters(model):
    """
    Tính số parameter cổ điển, lượng tử, và tổng số parameter của mô hình HybridQNN.
    
    Args:
        model (nn.Module): Mô hình HybridQNN.
    
    Returns:
        tuple: (classical_params, quantum_params, total_params)
    """
    classical_params = 0
    quantum_params = 0
    
    # Duyệt qua tất cả các tham số của mô hình
    for name, param in model.named_parameters():
        if 'quantum_layer' in name:
            # Tham số thuộc lớp lượng tử
            quantum_params += param.numel()
        else:
            # Tham số thuộc lớp cổ điển (classical_nn hoặc post_quantum_nn)
            classical_params += param.numel()
    
    total_params = classical_params + quantum_params
    
    return classical_params, quantum_params, total_params