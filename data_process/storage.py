"""Read large NPZ sample arrays without inflating them into RAM."""
from contextlib import contextmanager
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np


@contextmanager
def mapped_samples(source: Path):
    # NPZ members cannot be memory-mapped directly by np.load.
    with TemporaryDirectory(prefix=".samples-", dir=source.parent) as temporary:
        path = Path(temporary) / "X.npy"
        with ZipFile(source) as archive, archive.open("X.npy") as member, path.open("wb") as out:
            shutil.copyfileobj(member, out, length=1024 * 1024)
        samples = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            yield samples
        finally:
            samples._mmap.close()
