from pathlib import Path

import pytest
from conda_index.index import ChannelIndex

import changing_repodata


@pytest.fixture
def subdirs():
    return ("noarch", "linux-64")


@pytest.fixture
def index(subdirs):
    here = Path(__file__).parent
    channel_root = Path(__file__).parent / "conda-forge"
    subdirs = subdirs
    for subdir in subdirs:
        if not (channel_root / subdir / ".cache" / "cache.db").exists():
            print(f"Unpack {subdir}/cache.db.zst")
    index = ChannelIndex(
        channel_root,
        channel_name="conda-forge",
        subdirs=subdirs,
        threads=1,
        output_root=here / "output",
        cache_class=changing_repodata.DateLimitedCache,
    )
    return index
