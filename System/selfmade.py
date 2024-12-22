import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO

# ST-GCN Modules
def normalize_adj(adj):
    """Row-normalize the adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    return d_mat_inv.dot(adj)

class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STGCN, self).__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

    def forward(self, x, A):
        x = torch.einsum('nctv,vw->nctw', (x, A))  # Apply graph convolution
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        return x

# Initialize YOLO-Pose model
model = YOLO("yolo11n-pose.pt")

# Placeholder for adjacency matrix
num_nodes = 17  # Assuming 17 keypoints as output from YOLO-Pose
adjacency_matrix = np.eye(num_nodes)
adjacency_matrix = normalize_adj(adjacency_matrix)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

# Placeholder for input
in_channels = 2  # x and y coordinates for each keypoint
out_channels = 64  # Example output channels for GCN
stgcn = STGCN(in_channels, out_channels, num_nodes)

# Example inference loop
while True:
    results = model(1, show=True)  # Real-time video inference
    for result in results:
        keypoints = []
        for box in result.keypoints:
            keypoints.append(box.xy)  # Extract x, y coordinates of keypoints

        # Prepare input tensor
        keypoints = np.array(keypoints)  # (num_nodes, 2)
        keypoints = keypoints.T[np.newaxis, :, :, np.newaxis]  # Add batch, temporal, and channel dims
        input_tensor = torch.tensor(keypoints, dtype=torch.float32)

        # Forward pass through ST-GCN
        output = stgcn(input_tensor, adjacency_matrix)
        print("ST-GCN Output Shape:", output.shape)
