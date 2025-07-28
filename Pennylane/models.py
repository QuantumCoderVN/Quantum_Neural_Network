# models.py

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from config import N_QUBITS, N_LAYERS_ANSATZ

# --- Định nghĩa Mạch Lượng tử (Quantum Circuit) ---
# Sử dụng pennylane.numpy cho các hoạt động cần đạo hàm tự động trong QNode
dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Mapping đầu vào
    for i in range(N_QUBITS):
        qml.RY(pnp.pi * inputs[i], wires=i)

    # Entanglement
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # Ansatze (Parametrized Quantum Circuit)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # Đo lường
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# --- Định nghĩa Lớp Lượng tử (QuantumLayer) trong PyTorch ---
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers_ansatz=N_LAYERS_ANSATZ):
        super(QuantumLayer, self).__init__()
        self.qml_circuit = quantum_circuit
        # Trọng số của mạch lượng tử, được quản lý bởi PyTorch
        self.weights = nn.Parameter(torch.rand(n_layers_ansatz, n_qubits, 3, requires_grad=True))

    def forward(self, x):
        # x có batch_size x n_qubits
        # Chuyển đổi tensor PyTorch thành pennylane.numpy để truyền vào QNode
        # và sau đó chuyển lại kết quả về PyTorch tensor
        batch_out = []
        for i in range(x.shape[0]):
            # Đảm bảo inputs[i] là một tensor PyTorch đơn lẻ (không phải một batch)
            # và có gradient theo PyTorch
            q_out_list = self.qml_circuit(x[i], self.weights)
            q_out_tensor = torch.stack(q_out_list)
            batch_out.append(q_out_tensor)
        return torch.stack(batch_out)

# --- Định nghĩa Mô hình QNN Lai (HybridQNN) ---
class HybridQNN(nn.Module):
    def __init__(self, img_size, n_qubits=N_QUBITS, num_classes=10):
        super(HybridQNN, self).__init__()
        self.img_size = img_size
        self.n_qubits = n_qubits

        self.classical_nn = nn.Sequential(
            nn.Linear(img_size * img_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits) # Đầu ra của NN cổ điển là đầu vào cho mạch lượng tử
        )

        self.quantum_layer = QuantumLayer(n_qubits)

        self.post_quantum_nn = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        # Flatten ảnh
        x = x.view(x.size(0), -1)
        # Truyền qua lớp cổ điển
        x = self.classical_nn(x)
        # Truyền qua lớp lượng tử
        x = self.quantum_layer(x)
        # Truyền qua lớp cổ điển cuối cùng để phân loại
        x = self.post_quantum_nn(x)
        return x