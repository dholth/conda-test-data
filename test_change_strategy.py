#!/usr/bin/env python
"""
Detect changes per-package in different ways.
"""

import hashlib
import itertools
import json

import pytest

from changing_repodata import DateLimitedCache


def add_computed(db):
    """
    Add generated computed columns that may be helpful.

    (re-computes on each query returning these columns).
    """
    db.execute(
        "ALTER TABLE index_json ADD COLUMN IF NOT EXISTS name AS (json_extract(index_json, '$.name'))"
    )
    db.execute(
        "ALTER TABLE index_json ADD COLUMN IF NOT EXISTS sha256 AS (json_extract(index_json, '$.sha256')))"
    )


def package_names_in_stat_table():
    "ALTER TABLE stat ADD COLUMN IF NOT EXISTS name TEXT;"


#     sqlite> UPDATE stat
#    ...> SET name = index_json.name
#    ...> FROM (SELECT name, path FROM index_json) as index_json
#    ...> where stat.path = index_json.path;


def test_shard_sums(index, benchmark):
    sums_by_name: dict[str, hashlib._Hash] = {}
    cache: DateLimitedCache = index.cache_for_subdir("linux-64")  # type: ignore
    # only files in the upstream stage get included in the final index
    upstream = cache.upstream_stage

    # indexing's job is to make incoming sha256 and stat stage sha256 match by
    # extracting latest packages... we have also added a sha256 pulled from
    # index_json itself that we also added during the indexing phase.
    @benchmark
    def compute_sums():
        sums = {}
        for name, rows in itertools.groupby(
            cache.db.execute(
                """SELECT index_json.name, index_json.sha256, path
                FROM stat JOIN index_json USING (path) WHERE stat.stage = ?
                ORDER BY index_json.name, index_json.sha256""",
                (upstream,),
            ),
            lambda k: k[0],
        ):
            hasher = hashlib.sha256()
            for row in rows:
                name, sha256, path = row
                if name not in sums_by_name:
                    hasher.update((sha256 + path).encode())
            sums[name] = hasher.hexdigest()

        return sums

    assert compute_sums is not None


@pytest.mark.parametrize("algorithm", [hashlib.sha256, hashlib.blake2b])
def test_shard_sums_name_in_stat(index, benchmark, algorithm):
    sums_by_name: dict[str, hashlib._Hash] = {}
    cache: DateLimitedCache = index.cache_for_subdir("linux-64")  # type: ignore
    # only files in the upstream stage get included in the final index
    upstream = cache.upstream_stage

    # indexing's job is to make incoming sha256 and stat stage sha256 match by
    # extracting latest packages... we have also added a sha256 pulled from
    # index_json itself that we also added during the indexing phase.
    @benchmark
    def compute_sums():
        sums = {}
        for name, rows in itertools.groupby(
            cache.db.execute(
                """SELECT name, sha256, path
                FROM stat WHERE stat.stage = ?
                ORDER BY name, sha256""",
                (upstream,),
            ),
            lambda k: k[0],
        ):
            hasher = algorithm()
            for row in rows:
                name, sha256, path = row
                if name not in sums_by_name:
                    hasher.update((sha256 + path).encode())
            sums[name] = hasher.hexdigest()

        return sums

    assert compute_sums is not None


@pytest.mark.parametrize("subdir", ("noarch", "linux-64"))
@pytest.mark.benchmark(min_rounds=2)
def test_generate_whole_repodata(index, subdir, benchmark):
    @benchmark
    def generate():
        cache: DateLimitedCache = index.cache_for_subdir(subdir)  # type: ignore
        packages, packages_conda = cache.indexed_packages_by_timestamp()
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
        return json.dumps(repodata, separators=(":", ","), sort_keys=True)

    assert isinstance(generate, str)
