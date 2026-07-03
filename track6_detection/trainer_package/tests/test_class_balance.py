"""Minimal smoke test for src/class_balance.py's counting logic, runnable
without any real dataset or GPU (matches Hafnia reference repos' pattern
of shipping a tests/ dir alongside scripts/ and src/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_balance import compute_class_counts  # noqa: E402


def test_compute_class_counts_empty_dir(tmp_path):
    images_dir = tmp_path / "images" / "train"
    images_dir.mkdir(parents=True)
    counts = compute_class_counts(images_dir, nc=10)
    assert sum(counts.values()) == 0
    assert len(counts) == 10
