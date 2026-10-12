#!/usr/bin/env python3
"""
Fetch a small subset of the MicroBooNE public HDF5 samples (Zenodo).

Instead of downloading a whole 2-11 GB file, this streams the remote HDF5
over HTTP range requests and copies only:
  * the first N events, and
  * the tables needed for PFO-level particle ID (hits, Pandora PFOs,
    truth particles, energy deposits, event truth).
Optical tables (ophit/opflash) and wires are skipped.

Usage:
    pip install h5py "fsspec[http]" aiohttp requests numpy
    python fetch_microboone.py --sample nue --n-events 2000
    python fetch_microboone.py --sample inclusive --n-events 3000
    python fetch_microboone.py --sample nue --full   # whole file instead (~2 GB)

Data: MicroBooNE Collaboration, CC-BY 4.0.
  nue NoWire:       https://doi.org/10.5281/zenodo.7261921
  inclusive NoWire: https://doi.org/10.5281/zenodo.8370883
"""
import argparse
import os
import sys
import time

import numpy as np
import h5py

SAMPLES = {
    # name: (zenodo record, filename pattern, number of files)
    "nue": ("7261921", "nue_NoWire_{:02d}.h5", 20),
    "inclusive": ("8370883", "BNB_All_NoWire_{:02d}.h5", 18),
}

KEEP_TABLES = [
    "event_table",
    "hit_table",
    "pandoraHit_table",
    "pandoraPfp_table",
    "pandoraPrimary_table",
    "particle_table",
    "edep_table",
]

SEQ = "event_id.seq_cnt"


def url_for(sample, idx):
    rec, pattern, n = SAMPLES[sample]
    if not 0 <= idx < n:
        sys.exit(f"--file-index must be 0..{n - 1} for sample '{sample}'")
    name = pattern.format(idx)
    return f"https://zenodo.org/records/{rec}/files/{name}?download=1", name


def download_full(url, out_path):
    import requests

    tmp = out_path + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and r.status_code == 206 else "wb"
        done = have if mode == "ab" else 0
        t0 = time.time()
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                done += len(chunk)
                rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB  ({rate:.1f} MB/s)", end="")
    print()
    os.replace(tmp, out_path)


def rows_for_first_events(grp, n_events, event_ids_keep):
    """Return (n_rows, seq_cnt_rows_to_keep) for the first n_events of a table."""
    n_total = grp["event_id"].shape[0]
    if SEQ in grp:
        sc = grp[SEQ][:]
        # hep_hpc convention: [:, 0] = event sequence index, [:, 1] = row count.
        # Check which column is the count by summing.
        if sc[:, 1].sum() == n_total:
            seq_col, cnt_col = 0, 1
        elif sc[:, 0].sum() == n_total:
            seq_col, cnt_col = 1, 0
        else:
            seq_col = cnt_col = None
        if seq_col is not None:
            mask = sc[:, seq_col] < n_events
            # rows are stored contiguously in event order; keep the leading block
            k = int(np.argmin(mask)) if not mask.all() else len(mask)
            n_rows = int(sc[:k, cnt_col].sum())
            return n_rows, k
    # Fallback: scan event_id (slower, reads the whole column once)
    print("    (no usable seq_cnt; scanning event_id column)")
    ev = grp["event_id"][:]
    keep = np.array([tuple(e) in event_ids_keep for e in ev])
    if not keep.any():
        return 0, None
    n_rows = int(np.max(np.nonzero(keep)[0]) + 1)
    return n_rows, None


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def fetch_subset(url, out_path, n_events, block_size):
    import fsspec

    fs = fsspec.filesystem("http")
    t0 = time.time()
    with fs.open(url, "rb", block_size=block_size, cache_type="blockcache") as fobj, \
            h5py.File(fobj, "r") as src, \
            h5py.File(out_path + ".part", "w") as dst:

        copy_attrs(src, dst)
        print("Tables in remote file:", ", ".join(src.keys()))

        ev_all = src["event_table"]["event_id"]
        n_avail = ev_all.shape[0]
        n_events = min(n_events, n_avail)
        event_ids_keep = {tuple(e) for e in ev_all[:n_events]}
        print(f"Keeping first {n_events} of {n_avail} events")

        for name in KEEP_TABLES:
            if name not in src:
                print(f"  ! {name} not in file, skipping")
                continue
            g = src[name]
            n_total = g["event_id"].shape[0]
            n_rows, k_seq = rows_for_first_events(g, n_events, event_ids_keep)
            print(f"  {name}: {n_rows:,} / {n_total:,} rows", flush=True)
            og = dst.create_group(name)
            copy_attrs(g, og)
            for col in g.keys():
                ds = g[col]
                if col == SEQ and k_seq is not None:
                    data = ds[:k_seq]
                elif ds.shape and ds.shape[0] == n_total:
                    data = ds[:n_rows]
                else:
                    data = ds[()]
                kw = dict(compression="gzip", compression_opts=4) if np.ndim(data) else {}
                od = og.create_dataset(col, data=data, **kw)
                copy_attrs(ds, od)

            # sanity check: every kept row belongs to a kept event
            ev = og["event_id"][:]
            bad = sum(tuple(e) not in event_ids_keep for e in ev)
            if bad:
                print(f"    warning: {bad} rows from events outside the kept set")

    os.replace(out_path + ".part", out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({mb:.1f} MB) in {time.time() - t0:.0f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=SAMPLES, default="nue")
    p.add_argument("--file-index", type=int, default=0)
    p.add_argument("--n-events", type=int, default=2000)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--full", action="store_true", help="download the whole file instead")
    p.add_argument("--block-mb", type=int, default=8, help="HTTP range-request block size")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    url, name = url_for(a.sample, a.file_index)
    if a.full:
        out = os.path.join(a.out_dir, name)
        print(f"Downloading full file {name}")
        download_full(url, out)
    else:
        out = os.path.join(a.out_dir, name.replace(".h5", f"_first{a.n_events}.h5"))
        print(f"Streaming subset of {name}")
        fetch_subset(url, out, a.n_events, a.block_mb << 20)


if __name__ == "__main__":
    main()
