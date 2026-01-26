import numpy as np
import torch
import torch.nn as nn

class Conv2DFromScratch(nn.Module):
    """
    Custom 2D Convolution Layer implemented in pure NumPy.
    Wraps NumPy computation but maintains PyTorch nn.Module interface.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x):
        # Convert torch tensor to numpy
        x_np = x.detach().cpu().numpy()
        w_np = self.weight.detach().cpu().numpy()
        b_np = self.bias.detach().cpu().numpy()
        
        N, C_in, H, W = x_np.shape
        C_out = self.out_channels
        K = self.kernel_size
        
        # Apply padding using numpy
        if self.padding > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding)), mode='constant', constant_values=0)
            H = H + 2 * self.padding
            W = W + 2 * self.padding
        
        # Calculate output dimensions
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        # Initialize output
        out = np.zeros((N, C_out, H_out, W_out), dtype=x_np.dtype)
        
        # Perform convolution
        for n in range(N):
            for c_out in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + K
                        w_end = w_start + K
                        
                        # Extract patch: (C_in, K, K)
                        patch = x_np[n, :, h_start:h_end, w_start:w_end]
                        
                        # Get kernel: (C_in, K, K)
                        kernel = w_np[c_out, :, :, :]
                        
                        # Compute convolution: element-wise multiply and sum
                        out[n, c_out, i, j] = np.sum(patch * kernel) + b_np[c_out]
        
        # Convert back to torch tensor
        out_tensor = torch.from_numpy(out).to(x.device).to(x.dtype)
        
        return out_tensor

