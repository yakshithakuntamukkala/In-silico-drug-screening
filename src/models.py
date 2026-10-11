import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
try:
    import pennylane as qml
except ImportError:
    qml = None

# 1. Classical GNN (Standard Message Passing / GCN Backbone)
class ClassicalGNN(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = nn.Linear(16, 2) # Outputs [Solubility, Toxicity]
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# 2. Quantum-Enhanced GNN (Classical GNN + PennyLane Quantum Layer)
if qml is not None:
    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=[0, 1])
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

class QuantumEnhancedGNN(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 2) # Reduce to 2 features for 2 qubits
        self.weight_shape = {"weights": (2, 2, 3)}
        if qml is not None:
            self.qlayer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shape)
        self.fc_out = nn.Linear(2, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        
        # Pass through Quantum Layer
        if qml is not None:
            x = self.qlayer(x)
            
        return self.fc_out(x)

# 3. GAT (Graph Attention Network)
class GATModel(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.conv1 = GATConv(input_dim, 16, heads=2)
        self.conv2 = GATConv(16 * 2, 16)
        self.fc = nn.Linear(16, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# 4. GCN (Graph Convolutional Network)
class GCNModel(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = nn.Linear(16, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)