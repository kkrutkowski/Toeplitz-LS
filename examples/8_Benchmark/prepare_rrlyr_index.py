#!/usr/bin/env python3
"""Generate index_rrlyr.tsv for Benchmark 8 from OGLE LMC_RRlyr_table.txt."""
from pathlib import Path

TABLE_PATH = Path("/home/krutkowski/Pulpit/Toeplitz-LS/LMC_phot/LMC_RRlyr_table.txt")
OUTPUT_PATH = Path("/home/krutkowski/Pulpit/Toeplitz-LS/Toeplitz-ls/examples/8_Benchmark/index_rrlyr.tsv")

def main():
    rows = []
    with TABLE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            id_ = parts[0]
            p1 = float(parts[11])
            if p1 > 0:
                rows.append((id_, 1.0 / p1, p1))
    
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        out.write("ID\tF_1\tP_1\n")
        for id_, f1, p1 in rows:
            out.write(f"{id_}\t{f1:.15g}\t{p1:.15g}\n")
    print(f"Wrote {len(rows)} entries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
