import torch
from torch.nn import AvgPool2d, Linear, Flatten, Softmax
from torch.nn.functional import cross_entropy
import numpy as np


class OrientationEstimationModel(torch.nn.Module):
    def __init__(self, M=9, N=8):
        super().__init__()
        self.M=M
        self.N=N
        # Use EfficientNet for smaller model with similar performance to ResNet-101
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        # Average Pooling
        self.pool = AvgPool2d(3, stride=1)
        self.flatten = Flatten()
        # FC layers into MxN
        self.pre_orientation_buckets = Linear(64512, self.M*self.N)  
        # Final Softmax layer for each of M orientations
        self.softmaxes = Softmax(dim=2)


    def forward(self, x):
        stem = self.efficientnet.stem(x)
        mp = self.efficientnet.layers(stem)
        feature = self.efficientnet.features(mp)	
        p1 = self.pool(feature)
        o = self.pre_orientation_buckets(self.flatten(p1)).reshape([-1, self.M, self.N])
        buckets = self.softmaxes(o)
        return buckets

# Args:
# Each feature in bucket is G=360/N apart
# Buckets are offset (G/M)=360/(MN) apart
def create_gt(M, N, angles, use_degrees=True):
    # Create angle tensor
    _, m, n = np.ogrid[:len(angles), :M, :N]
    G = 360 / N if use_degrees else np.pi / N
    buckets = m*(G/M) + n*G
    i = np.argmax(
        np.cos(np.deg2rad(angles[:,np.newaxis,np.newaxis] - buckets)),
        axis=2)
    gt_buckets = torch.tensor(np.eye(N)[i])
    return gt_buckets


if __name__ == "__main__":
    obj = OrientationEstimationModel(N=20)
    output = obj(torch.rand((48, 3, 256, 256)))
    print(output.shape)
    print(torch.mean(output))
    gt = create_gt(8, 9, np.array([40, 271]))
    print(gt)
    data = torch.zeros([2, 8, 9])
    print(cross_entropy(data, gt))
