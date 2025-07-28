# config.py

import os
import torch

# --- Cấu hình thiết bị và Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KMP_DUPLICATE_LIB_OK = "TRUE" # Đặt biến môi trường để khắc phục lỗi OpenMP (OMP)

# --- Dữ liệu và Huấn luyện ---
BATCH_SIZE = 64
N_TRAIN_SAMPLES = 2000
N_TEST_SAMPLES = N_TRAIN_SAMPLES // 5
IMG_SIZE = 8
NUM_CLASSES = 10
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# --- Quantum Circuit ---
N_QUBITS = 4
N_LAYERS_ANSATZ = 1 # Số lớp của StronglyEntanglingLayers trong mạch lượng tử

# --- Thư mục và File đầu ra ---
RESULT_DIR = 'results'
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, 'hybrid_qnn_model.pth')
QUANTUM_CIRCUIT_PLOT_PATH = os.path.join(RESULT_DIR, 'quantum_circuit.png')
TRAINING_METRICS_PLOT_PATH = os.path.join(RESULT_DIR, 'training_metrics.png')
QUANTUM_WEIGHTS_PLOT_PATH = os.path.join(RESULT_DIR, 'quantum_weights_evolution.png')
PREDICTIONS_PLOT_PATH = os.path.join(RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(RESULT_DIR, 'classification_report.txt')