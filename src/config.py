import torch


def get_device() -> str:
    """Return the best available device: CUDA (Windows/Linux) > MPS (Mac) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
