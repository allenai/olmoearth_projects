"""fsspec-related utilities."""

import multiprocessing
import shutil

from fsspec.implementations.local import LocalFileSystem
from rslearn.utils.mp import star_imap_unordered
from upath import UPath


def copy_file(src_fname: UPath, dst_fname: UPath) -> None:
    """Copy a file from src to dst.

    This is mainly for use with multiprocessing.
    """
    if isinstance(dst_fname.fs, LocalFileSystem):
        # When copying to local filesystem, we should ensure parent directory is
        # created.
        dst_fname.parent.mkdir(parents=True, exist_ok=True)
    with src_fname.open("rb") as src:
        with dst_fname.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def get_relative_suffix(base_dir: UPath, fname: UPath) -> str:
    """Get the suffix of fname relative to base_dir.

    Args:
        base_dir: the base directory.
        fname: a filename within the base directory.

    Returns:
        the suffix on base_dir that would yield the given filename.
    """
    if not fname.path.startswith(base_dir.path):
        raise ValueError(
            f"filename {fname.path} must start with base directory {base_dir.path}"
        )
    suffix = fname.path[len(base_dir.path) :]
    if suffix.startswith("/"):
        suffix = suffix[1:]
    return suffix


def copy_files(src_dir: UPath, dst_dir: UPath, workers: int = 0) -> None:
    """Copy the source directory tree to the destination directory.

    Args:
        src_dir: the source directory to recursively copy from.
        dst_dir: the destination directory.
        workers: number of workers to use, or 0 to use main thread only. Workers are
            only used for the copying, not for discovering files to copy.
    """
    copy_file_jobs: list[dict] = []
    for src_fname in src_dir.rglob("*"):
        if src_fname.is_dir():
            continue

        dst_fname = dst_dir / get_relative_suffix(src_dir, src_fname)
        copy_file_jobs.append(
            dict(
                src_fname=src_fname,
                dst_fname=dst_fname,
            )
        )

    if workers == 0:
        for job in copy_file_jobs:
            copy_file(**job)
    else:
        p = multiprocessing.Pool(workers)
        outputs = star_imap_unordered(p, copy_file, copy_file_jobs)
        for _ in outputs:
            pass
        p.close()
