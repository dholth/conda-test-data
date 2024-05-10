#!/usr/bin/env python
"""
Detect changes per-package in different ways.
"""

import hashlib
import itertools
import json

import msgpack
import pytest
import zstandard

from changing_repodata import DateLimitedCache


def pack_package_record(record):
    """
    Convert hex checksums to bytes.
    """
    if sha256 := record.get("sha256"):
        record["sha256"] = bytes.fromhex(sha256)
    if md5 := record.get("md5"):
        record["md5"] = bytes.fromhex(md5)
    return record


def add_computed(db):
    """
    Add generated computed columns that may be helpful.

    (re-computes on each query returning these columns).
    """
    columns = set(row[1] for row in db.execute("pragma table_xinfo(index_json)"))
    if "name" not in columns:
        db.execute(
            "ALTER TABLE index_json ADD COLUMN IF NOT EXISTS name AS (json_extract(index_json, '$.name'))"
        )
    if "sha256" not in columns:
        db.execute(
            "ALTER TABLE index_json ADD COLUMN IF NOT EXISTS sha256 AS (json_extract(index_json, '$.sha256'))"
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

    add_computed(cache.db)

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
    """
    Test including a (populated by upsert) name in stat; see example in
    package_names_in_stat_table function.

    2.2x faster than test_shard_sums but only saves about 863ms on
    conda-forge/linux-64.
    """
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


@pytest.mark.benchmark(min_rounds=2)
def test_index_json_sums(index, benchmark):
    """
    Hash entire path, index_json contents to possibly find changes.
    """
    cache: DateLimitedCache = index.cache_for_subdir("linux-64")  # type: ignore
    # only files in the upstream stage get included in the final index
    upstream = cache.upstream_stage

    # indexing's job is to make incoming sha256 and stat stage sha256 match by
    # extracting latest packages... we have also added a sha256 pulled from
    # index_json itself that we also added during the indexing phase.
    @benchmark
    def compute_sums():
        sums = {}
        for path, rows in itertools.groupby(
            cache.db.execute(
                """SELECT index_json.name, path, index_json
                FROM stat JOIN index_json USING (path) WHERE stat.stage = ?
                ORDER BY index_json.name, index_json.path""",
                (upstream,),
            ),
            lambda k: k[0],
        ):
            hasher = hashlib.sha256()
            for row in rows:
                name, path, index_json = row
                hasher.update((path + index_json).encode())
            sums[name] = hasher.hexdigest()

        return sums

    assert compute_sums is not None


@pytest.mark.benchmark(min_rounds=1)
@pytest.mark.parametrize("level", (3,))
@pytest.mark.parametrize("codec", (json, msgpack))
def test_compress_shards(index, tmp_path, benchmark, level, codec):
    """
    Generate and write-to-disk unpatched repodata shards.
    """
    cache: DateLimitedCache = index.cache_for_subdir("linux-64")  # type: ignore
    # only files in the upstream stage get included in the final index
    upstream = cache.upstream_stage
    compressor = zstandard.ZstdCompressor(level=level)

    shard_path = tmp_path / "shards"
    shard_path.mkdir()

    def identity(record: dict):
        return record

    if codec is json:
        pack_record = identity
        dumps = lambda x: json.dumps(x).encode()
    else:
        pack_record = pack_package_record
        dumps = lambda x: msgpack.dumps(x)

    @benchmark
    def compress_shards():
        shards = {}
        for name, rows in itertools.groupby(
            cache.db.execute(
                """SELECT index_json.name, path, index_json
                FROM stat JOIN index_json USING (path) WHERE stat.stage = ?
                ORDER BY index_json.name, index_json.path""",
                (upstream,),
            ),
            lambda k: k[0],
        ):
            shard = {"packages": {}, "packages.conda": {}}
            hasher = hashlib.sha256()
            for row in rows:
                name, path, index_json = row
                assert path.endswith(
                    (".tar.bz2", ".conda")
                ), f"Unknown package filename {path}"
                hasher.update((path + index_json).encode())
                record = json.loads(index_json)
                key = "packages" if path.endswith(".tar.bz2") else "packages.conda"
                shard[key][path] = pack_record(record)

            reference_hash = hasher.hexdigest()

            (shard_path / f"{name}-{reference_hash}.{codec.__name__}.zst").write_bytes(
                compressor.compress(dumps(shard))  # type: ignore
            )

            shards[name] = shard

        return shards

    assert compress_shards is not None

    original_size = sum(p.stat().st_size for p in shard_path.glob("*.zst"))
    print(f"{original_size} bytes with {codec.__name__}x{level}")

    compress_dict = zstandard.train_dictionary(
        2**16,
        [zstandard.decompress(p.read_bytes()) for p in shard_path.glob("*.zst")],
    )

    assert compress_dict

    compress2 = zstandard.ZstdCompressor(dict_data=compress_dict, level=level)
    dir2 = tmp_path / "dict_compression"
    dir2.mkdir()

    for p in shard_path.glob("*.zst"):
        (dir2 / p.name).write_bytes(
            compress2.compress(zstandard.decompress(p.read_bytes()))
        )

    dict_compression_size = sum(p.stat().st_size for p in dir2.glob("*.zst"))
    print(
        "Did we save space?", (original_size / dict_compression_size) > 1.1
    )  # no we did not
