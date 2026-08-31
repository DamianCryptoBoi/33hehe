import subprocess
import sys


def test_pinned_torch_supports_numpy_interop():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import numpy as np; import torch; "
                "array = np.array([1.0], dtype=np.float32); "
                "assert torch.from_numpy(array).numpy().tolist() == [1.0]"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Failed to initialize NumPy" not in result.stderr
