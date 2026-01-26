import numpy as np
import torch
import torch.nn as nn

class MaxPool2DFromScratch(nn.Module):
    """
    Custom 2D Max Pooling Layer implemented in pure NumPy.
    Maintains PyTorch nn.Module interface for seamless integration.
    """
    
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        # Convert torch tensor to numpy
        x_np = x.detach().cpu().numpy()
        
        N, C, H, W = x_np.shape
        K = self.kernel_size
        S = self.stride
        
        # Calculate output dimensions
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1
        
        # Initialize output array
        out = np.zeros((N, C, H_out, W_out), dtype=x_np.dtype)
        
        # Perform max pooling using numpy
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Calculate window boundaries
                        h_start = i * S
                        w_start = j * S
                        h_end = h_start + K
                        w_end = w_start + K
                        
                        # Extract window from input
                        window = x_np[n, c, h_start:h_end, w_start:w_end]
                        
                        # Find maximum value in window
                        out[n, c, i, j] = np.max(window)
        
        # Convert back to torch tensor, preserving device and dtype
        out_tensor = torch.from_numpy(out).to(x.device).to(x.dtype)
        
        return out_tensor