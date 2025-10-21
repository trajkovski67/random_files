#!/usr/bin/env python3
"""
Compute attenuation factor including charge-density and atom-count suppression.

A = 1 / (1 + α(|q|/R_eff)^β)
S = 1 / (1 + exp[-k (N - N0)])
A' = A * S

Usage:
    ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K]

Example:
    ./att_factor 3.0 3.0 -1 --n0 10 --k 0.5
"""

import sys
import math
from pathlib import Path


def parse_cpcm(path: Path):
    """Extract (natoms, volume_bohr3) from .cpcm file."""
    natoms = None
    volume = None
    with open(path) as f:
        for line in f:
            if "# Number of atoms" in line:
                # Next line contains the number of atoms
                prev = line.strip().split()
                if len(prev) >= 1:
                    try:
                        natoms = int(prev[0])
                    except ValueError:
                        pass
            if "Number of atoms" in line and "#" not in line:
                try:
                    natoms = int(line.strip().split()[0])
                except ValueError:
                    pass
            if "Volume" in line:
                try:
                    volume = float(line.strip().split()[0])
                except ValueError:
                    pass
            if natoms is not None and volume is not None:
                break
    if natoms is None or volume is None:
        raise ValueError(f"Failed to parse atoms/volume in {path}")
    return natoms, volume


def compute_reff(volume_bohr3: float) -> float:
    """Convert CPCM volume (bohr³) to effective radius (Å)."""
    bohr_to_ang = 0.529177
    v_ang3 = volume_bohr3 * (bohr_to_ang ** 3)
    return (3.0 * v_ang3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def attenuation(alpha: float, beta: float, q: float, reff: float) -> float:
    """Charge-density based attenuation."""
    return 1.0 / (1.0 + alpha * (abs(q) / reff) ** beta)


def size_factor(natoms: int, n0: float = 10.0, k: float = 0.5) -> float:
    """Size-dependent attenuation term."""
    return 1.0 / (1.0 + math.exp(-k * (natoms - n0)))


def main():
    if len(sys.argv) < 4:
        print("Usage: ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K]")
        sys.exit(1)

    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    charge = float(sys.argv[3])
    n0 = 10.0
    k = 0.5

    if "--n0" in sys.argv:
        n0 = float(sys.argv[sys.argv.index("--n0") + 1])
    if "--k" in sys.argv:
        k = float(sys.argv[sys.argv.index("--k") + 1])

    root = Path(".")
    files = list(root.rglob("inp.cpcm"))
    if not files:
        sys.exit("No inp.cpcm files found.")

    for f in sorted(files):
        try:
            natoms, volume = parse_cpcm(f)
            reff = compute_reff(volume)
            A = attenuation(alpha, beta, charge, reff)
            S = size_factor(natoms, n0, k)
            Atot = A * S
            foldername = f.parents[1].name if f.parent.name == "molalign" else f.parent.name
            print(f"{foldername:30s}  A'={Atot:.4f}  (A={A:.4f}, N={natoms}, R_eff={reff:.3f})")
        except Exception as e:
            print(f"[warn] {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

