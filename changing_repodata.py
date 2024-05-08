#!/usr/bin/env python
"""
Generate changing repodata.json based on advancing mtimes.
"""

from pathlib import Path
from conda_index.index import ChannelIndex, sqlitecache
import json
import datetime
from conda_index.utils import CONDA_PACKAGE_EXTENSION_V1, CONDA_PACKAGE_EXTENSION_V2

import logging

log = logging.getLogger(__name__)


class DateLimitedCache(sqlitecache.CondaIndexCache):
    upstream_stage = "clone"

    def indexed_packages_by_timestamp(
        self, early_timestamp=0, late_timestamp=1800000000
    ):
        """
        Return "packages" and "packages.conda" values from the cache but only if mtime is between early_timestamp and late_timestamp.
        """
        new_repodata_packages = {}
        new_repodata_conda_packages = {}

        # load cached packages
        for row in self.db.execute(
            """
            SELECT path, index_json FROM stat JOIN index_json USING (path)
            WHERE stat.stage = ?
            AND mtime BETWEEN ? AND ?
            ORDER BY path
            """,
            (self.upstream_stage, early_timestamp, late_timestamp),
        ):
            path, index_json = row
            index_json = json.loads(index_json)
            if path.endswith(CONDA_PACKAGE_EXTENSION_V1):
                new_repodata_packages[path] = index_json
            elif path.endswith(CONDA_PACKAGE_EXTENSION_V2):
                new_repodata_conda_packages[path] = index_json
            else:
                log.warn("%s doesn't look like a conda package", path)

        return new_repodata_packages, new_repodata_conda_packages


def main():
    here = Path(__file__).parent
    channel_root = Path(__file__).parent / "conda-forge"
    subdirs = ("noarch", "linux-64")
    index = ChannelIndex(
        channel_root,
        channel_name="conda-forge",
        subdirs=subdirs,
        threads=1,
        output_root=here / "output",
        cache_class=DateLimitedCache,
    )

    linux_64_max_timestamp = (
        index.cache_for_subdir("linux-64")
        .db.execute("SELECT MAX(mtime) FROM stat WHERE stage='clone'")
        .fetchall()[0][0]
    )
    print("max timestamp", linux_64_max_timestamp)
    delta = datetime.timedelta(days=7).total_seconds()
    steps = 8

    for step in range(steps - 1, -1, -1):
        max_timestamp = linux_64_max_timestamp + (-delta * step)
        when = datetime.datetime.fromtimestamp(max_timestamp).replace(
            tzinfo=datetime.timezone.utc
        )
        print(f"Until {when}...")
        for subdir in subdirs:
            if not (channel_root / subdir / ".cache" / "cache.db").exists():
                print(f"unpack {subdir}/cache.db.zst")
                continue
            cache: DateLimitedCache = index.cache_for_subdir(subdir)  # type: ignore
            packages, packages_conda = cache.indexed_packages_by_timestamp(
                late_timestamp=max_timestamp
            )
            repodata = {
                "info": {
                    "subdir": subdir,
                },
                "packages": packages,
                "packages.conda": packages_conda,
                "removed": [],  # can be added by patch/hotfix process
                "repodata_version": 1,
            }
            print(
                f"{subdir}/repodata.json has {len(packages)+len(packages_conda)} packages"
            )
            output = index.output_root / subdir
            if not output.exists():
                output.mkdir(parents=True, exist_ok=True)
            Path(output / "repodata.json").write_text(
                json.dumps(repodata, separators=(":", ","), sort_keys=True)
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
