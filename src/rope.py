import torch
from einops import rearrange

def generate_angles_2d(H,W,D, device='cpu', freq=None):
    """
    Generates a 3D frequency field for 2D Rotary Positional Embeddings.
    - H: Height of the feature map.
    - W: Width of the feature map.
    - D: Embedding Dimension (must be even).
    - freq: Optional precomputed frequency tensor for the embedding dimension.
    """
    assert D % 2 == 0, "Embedding Dimension must be even!"
    freq = torch.tensor([10000**(-2*i/D) for i in range(int(D/2))], device=device) if freq is None else freq
    pos = torch.outer(torch.linspace(-1, 1, steps=H, device=device),torch.linspace(-1, 1, steps=W, device=device))
    freq_tensor = torch.einsum("ij,k->ijk", pos, freq) # outer product
    return freq_tensor

def generate_angles_1d(N, D, device='cpu', freq=None):
    """
    1d variation of generate_angles_2d
    """
    assert D % 2 == 0, "Embedding Dimension must be even!"
    freq = torch.tensor([10000**(-2*i/D) for i in range(int(D/2))], device=device) if freq is None else freq
    pos = torch.linspace(-1, 1, steps=N, device=device)
    freq_tensor = torch.einsum("i,j->ij", pos, freq) # outer product
    return freq_tensor

def apply_angles_2d(x, f):
    """
    Applies the 2D Rotary Positional Embeddings to the input tensor.
    - x: Input tensor of shape (B, H, W, D)
    - f: Frequency tensor of shape (H, W, D/2)
    Rotates each pair of dimensions in the last dimension via orthogonal 2D matrix multiplication.
    """
    x_reshaped = rearrange(x, "B H W (D p) -> B H W D p", p=2)
    real = x_reshaped[..., 0]
    imag = x_reshaped[..., 1]
    cosines, sines = f.cos(), f.sin()
    # r , i -> rcos-isin , rsin icos
    rot_real = real * cosines - imag * sines
    rot_imag = real * sines + imag * cosines
    rot_full = torch.concat((rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)), dim=-1)
    return rearrange(rot_full, "B H W D p -> B H W (D p)", p=2)

def apply_angles_1d(x, f):
    """
    1d variation of apply_angles_2d
    """
    x_reshaped = rearrange(x, "... (D p) -> ... D p", p=2)
    real = x_reshaped[..., 0]
    imag = x_reshaped[..., 1]
    cosines, sines = f.cos(), f.sin()
    # r , i -> rcos-isin , rsin icos
    rot_real = real * cosines[:real.shape[-2], :] - imag * sines[:real.shape[-2], :]
    rot_imag = real * sines[:real.shape[-2], :] + imag * cosines[:real.shape[-2], :]
    rot_full = torch.concat((rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)), dim=-1)
    return rearrange(rot_full, "... D p -> ... (D p)", p=2)

# Sanity Check :)
if __name__ == "__main__":
    x = apply_angles_1d(torch.randn(1,4,43,768), generate_angles_1d(64,768))
    assert x.isnan().sum() == False
    print(x.shape)
    x = apply_angles_2d(torch.randn(1,64,64,768), generate_angles_2d(64,64,768))
    assert x.isnan().sum() == False
    print(x.shape)