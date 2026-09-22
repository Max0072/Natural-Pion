"""harness/local_cache.py: staging a large file to local disk before use.

Pins the behaviour the wedge fix (docs/CLUSTER.md, "rtx6002 wedges too",
2026-09-22) depends on: staging is skipped below the size threshold and when
disabled, copies once and reuses the copy, and never raises -- a training run
must get a usable path back even if the local cache turns out to be broken.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from harness.local_cache import _MIN_SIZE_TO_STAGE, stage_locally


def _make(path, size):
    with open(path, "wb") as f:
        f.truncate(size)


def test_small_file_is_not_staged(tmp_path):
    """Below the threshold, staging is pure overhead for no benefit -- skip it."""
    src = tmp_path / "small.bin"
    _make(src, 1024)
    out = stage_locally(str(src), str(tmp_path / "cache"))
    assert out == str(src)
    assert not (tmp_path / "cache").exists()


def test_empty_cache_dir_disables_staging():
    """`local_cache_dir = ""` is the off switch, and must be a true no-op."""
    assert stage_locally("/anything", "") == "/anything"


def test_large_file_is_copied_and_reused(tmp_path):
    """The one behaviour the wedge fix depends on: read from local disk after."""
    src = tmp_path / "big.bin"
    _make(src, _MIN_SIZE_TO_STAGE + 1)
    cache = tmp_path / "cache"

    out1 = stage_locally(str(src), str(cache))
    assert out1 == str(cache / "big.bin")
    assert os.path.getsize(out1) == _MIN_SIZE_TO_STAGE + 1
    assert not (cache / "big.bin.partial").exists()

    # Second call: idempotent, a byte-size match short-circuits the copy.
    with mock.patch("shutil.copyfile", side_effect=AssertionError("should not copy again")):
        out2 = stage_locally(str(src), str(cache))
    assert out2 == out1


def test_stale_or_wrong_size_copy_is_replaced(tmp_path):
    """A local file that does not match the source's size is not trusted."""
    src = tmp_path / "big.bin"
    _make(src, _MIN_SIZE_TO_STAGE + 1)
    cache = tmp_path / "cache"
    cache.mkdir()
    _make(cache / "big.bin", 10)  # a stale or partial leftover

    out = stage_locally(str(src), str(cache))
    assert os.path.getsize(out) == _MIN_SIZE_TO_STAGE + 1


def test_missing_source_is_left_for_the_real_open_to_report(tmp_path):
    """No exception here -- `TokenCorpus` raises its own, clearer error."""
    out = stage_locally(str(tmp_path / "does-not-exist.bin"), str(tmp_path / "cache"))
    assert out == str(tmp_path / "does-not-exist.bin")


def test_unwritable_cache_dir_falls_back_to_the_source_path(tmp_path):
    """A broken or full local disk must not fail the run over an optimisation."""
    src = tmp_path / "big.bin"
    _make(src, _MIN_SIZE_TO_STAGE + 1)
    with mock.patch("pathlib.Path.mkdir", side_effect=OSError("no space left on device")):
        out = stage_locally(str(src), str(tmp_path / "cache"))
    assert out == str(src)


def test_concurrent_callers_copy_once(tmp_path):
    """The lock: two callers racing to stage the same file must not corrupt it.

    A real race is two processes; here two threads exercise the same flock
    path, which is what `stage_locally` actually takes.
    """
    import threading

    src = tmp_path / "big.bin"
    _make(src, _MIN_SIZE_TO_STAGE + 1)
    cache = tmp_path / "cache"
    results = []

    def run():
        results.append(stage_locally(str(src), str(cache)))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1
    assert os.path.getsize(results[0]) == _MIN_SIZE_TO_STAGE + 1
    assert not (cache / "big.bin.partial").exists()
