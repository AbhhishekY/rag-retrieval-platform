"""RAG retrieval platform."""
__version__ = "0.1.0"

import sys as _sys

class _StderrFilter:
    """Drop the noisy ResourceTracker cleanup error from multiprocess/ONNX."""
    _NOISE = ("ResourceTracker", "_recursion_count", "resource_tracker.py")

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s: str) -> int:
        if any(tok in s for tok in self._NOISE):
            return len(s)
        return self._wrapped.write(s)

    def flush(self) -> None:
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

if not isinstance(_sys.stderr, _StderrFilter):
    _sys.stderr = _StderrFilter(_sys.stderr)
