"""Choose the PyTorch device (GPU or CPU) from a config value."""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a device name from a config file into a ``torch.device``.

    - ``"auto"``: the GPU if CUDA is available, otherwise the CPU.
    - ``"cpu"``: always the CPU (e.g. for CPU latency measurements).
    - ``"cuda"`` / ``"cuda:0"``: the GPU; fails loudly if there is none, so a broken
      CUDA install is noticed instead of silently training on the CPU.
    """
    name = requested.strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {requested!r} was requested, but PyTorch does not see a CUDA GPU. "
                "Use device: auto or cpu, or reinstall the CUDA build of PyTorch."
            )
        return torch.device(name)
    raise ValueError(f"Unknown device {requested!r}; expected 'auto', 'cpu' or 'cuda[:index]'.")
