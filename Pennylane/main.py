# main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Nhập các cấu hình
from config import (
    DEVICE, KMP_DUPLICATE_LIB_OK, RESULT_DIR, LOG_FILE, NUM_EPOCHS, LEARNING_RATE,
    IMG_SIZE, N_QUBITS, NUM_CLASSES, MODEL_SAVE_PATH
)
# Nhập các hàm tải dữ liệu
from data_loader import get_mnist_data_loaders
# Nhập các lớp mô hình
from models import HybridQNN, quantum_circuit
# Nhập các hàm tiện ích
from utils import (
    setup_results_directory, write_log, plot_quantum_circuit,
    plot_predictions, plot_training_metrics, plot_quantum_weights_evolution,
    evaluate_and_report, count_parameters
)

def main():
    # --- Đặt biến môi trường ---
    os.environ["KMP_DUPLICATE_LIB_OK"] = KMP_DUPLICATE_LIB_OK

    # --- Cài đặt thư mục kết quả ---
    setup_results_directory()
    write_log(f"Using device: {DEVICE}")
    print(f"Using device: {DEVICE}")
    if DEVICE.type != "cpu":
        torch.cuda.empty_cache()

    # --- Tải và chuẩn bị dữ liệu ---
    train_loader, test_loader = get_mnist_data_loaders()

    # --- Định nghĩa và Khởi tạo Mô hình ---
    model = HybridQNN(IMG_SIZE, N_QUBITS, NUM_CLASSES).to(DEVICE)
    print("\n--- Model Architecture ---")
    print(model)
    write_log(f"\n--- Model Architecture ---\n{model}")

    # --- In số parameter ---
    classical_params, quantum_params, total_params = count_parameters(model)
    params_message = (f"Number of parameters:\n"
                     f"  - Classical parameters: {classical_params}\n"
                     f"  - Quantum parameters: {quantum_params}\n"
                     f"  - Total parameters: {total_params}")
    print(params_message)
    write_log(params_message)

    # --- Vẽ và lưu kiến trúc mạch lượng tử ---
    plot_quantum_circuit(quantum_circuit)

    # --- Cấu hình Huấn luyện ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Khởi tạo danh sách lưu trữ metrics ---
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    quantum_weights_history = [] # Lưu trữ trọng số lượng tử để vẽ biểu đồ

    print("\n--- Training Started ---")
    write_log("\n--- Training Started ---")

    for epoch in range(NUM_EPOCHS):
        # Lưu giá trị trọng số lượng tử của epoch hiện tại
        quantum_weights_history.append(model.quantum_layer.weights.detach().cpu().numpy().copy())

        # --- Training ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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

        # --- Validation (Testing) ---
        model.eval()
        test_loss_epoch = 0.0
        correct_test = 0
        total_test = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss_epoch += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss_epoch / len(test_loader)
        test_accuracy = correct_test / total_test
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)

        log_message = (f"Epoch {epoch+1}/{NUM_EPOCHS}: "
                       f"Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, "
                       f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
        print(log_message)
        write_log(log_message)

    print("--- Training Complete ---")
    write_log("--- Training Complete ---")

    # --- Lưu mô hình ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")
    write_log(f"Model weights saved to {MODEL_SAVE_PATH}")

    # --- Vẽ và Lưu biểu đồ kết quả ---
    plot_predictions(model, test_loader, DEVICE)
    plot_training_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, NUM_EPOCHS)
    plot_quantum_weights_evolution(quantum_weights_history, NUM_EPOCHS)
    evaluate_and_report(all_labels, all_preds)

if __name__ == "__main__":
    main()