import pytest
import torch

from gesture_bc.utils.device import resolve_device


def test_cpu_is_always_available():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_uses_gpu_only_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_explicit_cuda_without_gpu_fails_loudly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_device_name_ignores_case_and_spaces():
    assert resolve_device(" CPU ") == torch.device("cpu")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")
