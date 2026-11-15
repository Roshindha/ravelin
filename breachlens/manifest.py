"""B13a: layout manifests of a training run and the checker that refuses FINAL / transfer / LAB-DESIGN ids and streams.

A manifest lists every layout id a run can load: the training set handed to the env sampler and the validation set handed to the evaluator.
`check_manifest` fails (ManifestError) on: an unknown or mismatched set pair; any FINAL / transfer id (`pool.is_final`); any LAB-DESIGN id or stream L;
labyrinth; for the TUNE pair any id outside stream T; for the DEV pair any id outside the DEV part of stream F; duplicate ids; train/validation overlap;
and ids that are not exactly the declared set of the loader (design_records / dev records).
"""
from __future__ import annotations

import json

from . import layouts as Y
from . import pool as P

PAIRS = {"TUNE-train": "TUNE-val", "DEV-train": "DEV-validation"}


class ManifestError(ValueError):
    pass


def build_manifest(run_name: str, train_set: str, val_set: str, train_ids, val_ids, **extra) -> dict:
    return {"run_name": run_name, "train_set": train_set, "val_set": val_set, "train_ids": sorted(set(train_ids)), "val_ids": sorted(set(val_ids)),
            "n_train_listed": len(list(train_ids)), "n_val_listed": len(list(val_ids)),
            "scope": "every layout id the env sampler / evaluator of this run was given (a superset of what was drawn)", **extra}


def _expected(train_set: str, val_set: str) -> tuple[set, set]:
    if train_set == "TUNE-train":
        return ({r["id"] for r in P.design_records("TUNE-train", "train")}, {r["id"] for r in P.design_records("TUNE-val", "validation")})
    recs = P.dev_records
    return ({r["id"] for r in recs("train")}, {r["id"] for r in recs("validation")})


def check_manifest(m: dict, strict_sets: bool = True) -> None:
    ts, vs = m.get("train_set"), m.get("val_set")
    if PAIRS.get(ts) != vs:
        raise ManifestError(f"unknown / mismatched set pair {ts!r} -> {vs!r}")
    tr, va = list(m["train_ids"]), list(m["val_ids"])
    if len(set(tr)) != len(tr) or len(set(va)) != len(va):
        raise ManifestError("duplicate layout ids")
    if set(tr) & set(va):
        raise ManifestError(f"train / validation overlap: {sorted(set(tr) & set(va))[:3]}")
    for i in tr + va:
        fam, seed = P.parse_id(i)
        stream = Y.stream_of(seed)
        if P.is_final(fam, seed):
            raise ManifestError(f"FINAL / transfer layout in manifest: {i}")
        if stream == "L":
            raise ManifestError(f"LAB-DESIGN (stream L) layout in manifest: {i}")
        if fam == "labyrinth":
            raise ManifestError(f"labyrinth layout in manifest: {i}")
        if ts == "TUNE-train" and stream != "T":
            raise ManifestError(f"layout outside stream T in a TUNE manifest: {i} (stream {stream})")
        if ts == "DEV-train" and stream != "F":
            raise ManifestError(f"layout outside the DEV part of stream F in a DEV manifest: {i} (stream {stream})")
    if strict_sets:
        etr, eva = _expected(ts, vs)
        if set(tr) != etr or set(va) != eva:
            raise ManifestError("manifest ids are not exactly the declared sets of the loader")


def check_manifest_file(path) -> dict:
    m = json.loads(open(path).read())
    check_manifest(m)
    return m
