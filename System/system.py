import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np

class Graph:
    """
    Define the graph structure and adjacency matrix.
    """
    def __init__(self, layout='coco', strategy='spatial'):
        self.A = self.get_adjacency_matrix(layout, strategy)

    def get_adjacency_matrix(self, layout, strategy):
        # Example for COCO layout
        if layout == 'coco':
            num_nodes = 17
            self.num_nodes = num_nodes
            adjacency = np.eye(num_nodes)
            # Example: Define adjacency relationships for COCO keypoints
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Left arm
                (5, 6), (6, 7), (7, 8),          # Right arm
                (9, 10), (10, 11), (11, 12),     # Left leg
                (13, 14), (14, 15), (15, 16)     # Right leg
            ]
            for i, j in edges:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
            return adjacency
        else:
            raise ValueError(f"Unsupported layout: {layout}")

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GraphConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1), bias=False)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super(st_gcn, self).__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size[0], 1),
                      stride=(stride, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)

class ModernSTGCN(nn.Module):
    def __init__(self, in_channels, num_classes, graph_args, edge_importance_weighting=True):
        super(ModernSTGCN, self).__init__()
        # Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            st_gcn(256, 256, kernel_size, 1)
        ])

        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Classification layer
        self.cls = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x).view(N, V, C, T).permute(0, 2, 3, 1)

        # Forward through GCN layers
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = self.cls(x).view(x.size(0), -1)
        return x

# Example usage with YOLO integration
def preprocess_pose_data(yolo_results):
    """
    Convert YOLO pose data to a format suitable for ST-GCN.
    Input:
        yolo_results: YOLO detection results containing keypoints
    Output:
        Tensor of shape (N, C, T, V), where:
        - N: Batch size (number of detected persons)
        - C: Channels (keypoint coordinates)
        - T: Temporal dimension (sequence length, for simplicity = 1 here)
        - V: Number of graph nodes (keypoints)
    """
    keypoints = []
    for result in yolo_results:
        for box in result.boxes:
            if hasattr(box, 'keypoints') and box.keypoints is not None:
                keypoints.append(box.keypoints)

    if len(keypoints) == 0:
        raise ValueError("No keypoints found in YOLO results.")

    # Stack keypoints into a single tensor
    keypoints = torch.stack(keypoints).unsqueeze(2)  # Add temporal dimension
    return keypoints

# Load YOLO model
yolo_model = YOLO("yolo11n-pose.pt")

# Load ST-GCN model
stgcn_model = ModernSTGCN(in_channels=2, num_classes=10, graph_args={'layout': 'coco', 'strategy': 'spatial'})
stgcn_model.eval()

# Predict with YOLO and process data
yolo_results = yolo_model(1, show=True)
try:
    pose_data = preprocess_pose_data(yolo_results)
    # Predict with ST-GCN
    output = stgcn_model(pose_data)
    print("Action Recognition Output:", output)
except ValueError as e:
    print("Error during preprocessing:", e)
