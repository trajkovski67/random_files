#!/usr/bin/env python3
"""
Compute attenuation factor including charge‑density, size, and dipole‑based corrections
(active mainly for small systems).

Model:
    μ  = |Σ q_i (r_i − r_geom)|          # geometric‑centre dipole magnitude  (Å·e)
    μ̃ = μ / R_eff
    A  = 1 / [1 + α (|q| / R_eff)^β]
    S  = 1 / [1 + exp(−k (N − N0))]
    A' = A * [ S + (1 − S) * c * μ̃ ]    # dipole correction fades with size

Expected directory structure:
    molecule/
      ├── molalign/inp.cpcm   → geometry file (natoms, volume, coords)
      └── alpb/charges        → partial charges, one per atom

Usage:
    ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K] [--c C]
Example:
    ./att_factor 3.0 3.0 -1 --n0 10 --k 0.5 --c 0.6
"""

import sys, math
from pathlib import Path


# ────────────────────────────────────────
#  Parsing input files
# ────────────────────────────────────────
def parse_cpcm(path: Path):
    """Extract (natoms, volume_bohr3, coords) from inp.cpcm."""
    natoms, volume = None, None
    coords = []
    reading_coords = False
    with open(path) as f:
        for line in f:
            # number of atoms & volume
            if "# Number of atoms" in line or "Number of atoms" in line:
                try:
                    natoms = int(line.strip().split()[0])
                except Exception:
                    pass
            if "Volume" in line:
                try:
                    volume = float(line.strip().split()[0])
                except Exception:
                    pass

            # coordinate section
            if "# CARTESIAN COORDINATES" in line:
                reading_coords = True
                continue
            if reading_coords:
                if line.strip().startswith("#") or not line.strip():
                    if coords:
                        break
                    else:
                        continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    x, y, z = map(float, parts[:3])
                    coords.append((x, y, z))
                    if natoms and len(coords) >= natoms:
                        break
                except Exception:
                    continue

    if natoms is None or volume is None:
        raise ValueError(f"Cannot parse natoms/volume from {path}")
    if len(coords) != natoms:
        raise ValueError(f"Coordinate count {len(coords)} != natoms {natoms}")
    return natoms, volume, coords


def read_partial_charges(base_dir: Path, natoms: int):
    """Read partial charges from alpb/charges (space/newline separated)."""
    charges_path = base_dir / "alpb" / "charges"
    if not charges_path.exists():
        raise FileNotFoundError(f"No charges file found: {charges_path}")
    text = charges_path.read_text().strip()
    charges = [float(x) for x in text.split()]
    if len(charges) != natoms:
        raise ValueError(f"Charge count ({len(charges)}) != natoms ({natoms}) in {charges_path}")
    return charges


# ────────────────────────────────────────
#  Physics helper functions
# ────────────────────────────────────────
def compute_reff(volume_bohr3: float) -> float:
    """Convert CPCM volume (bohr³) → effective radius (Å)."""
    bohr_to_ang = 0.529177
    v_ang3 = volume_bohr3 * (bohr_to_ang ** 3)
    return (3.0 * v_ang3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def compute_dipole(charges, coords):
    """
    Compute geometric‑centre dipole magnitude (Å·e).
    Works for neutral and charged systems; measures internal charge separation.
    """
    bohr_to_ang = 0.529177
    n = len(coords)
    cx = sum(x for x, _, _ in coords) / n
    cy = sum(y for _, y, _ in coords) / n
    cz = sum(z for _, _, z in coords) / n

    mu_vec = [0.0, 0.0, 0.0]
    for q, (x, y, z) in zip(charges, coords):
        dx, dy, dz = x - cx, y - cy, z - cz
        mu_vec[0] += q * dx * bohr_to_ang
        mu_vec[1] += q * dy * bohr_to_ang
        mu_vec[2] += q * dz * bohr_to_ang

    mu = math.sqrt(sum(v * v for v in mu_vec))
    return mu


def attenuation(alpha, beta, q, reff):
    """Base charge‑density attenuation (0‑1)."""
    return 1.0 / (1.0 + alpha * (abs(q) / reff) ** beta)


def size_factor(natoms, n0=10.0, k=0.5):
    """Sigmoid size factor (0‑1)."""
    return 1.0 / (1.0 + math.exp(-k * (natoms - n0)))


# ────────────────────────────────────────
#  Main routine
# ────────────────────────────────────────
def main():
    if len(sys.argv) < 4:
        print("Usage: ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K] [--c C]")
        sys.exit(1)

    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    charge = float(sys.argv[3])
    n0 = 10.0
    k = 0.5
    c = 0.6   # dipole weighting coefficient

    if "--n0" in sys.argv:
        n0 = float(sys.argv[sys.argv.index("--n0") + 1])
    if "--k" in sys.argv:
        k = float(sys.argv[sys.argv.index("--k") + 1])
    if "--c" in sys.argv:
        c = float(sys.argv[sys.argv.index("--c") + 1])

    root = Path(".")
    cpcm_files = list(root.rglob("molalign/inp.cpcm"))
    if not cpcm_files:
        sys.exit("No molalign/inp.cpcm files found.")

    for cfile in sorted(cpcm_files):
        base = cfile.parents[1]  # molecule directory
        try:
            natoms, volume, coords = parse_cpcm(cfile)
            reff = compute_reff(volume)
            charges = read_partial_charges(base, natoms)
            mu = compute_dipole(charges, coords)
            mu_tilde = mu / reff

            A = attenuation(alpha, beta, charge, reff)
            S = size_factor(natoms, n0, k)

            # dipole correction mainly for small species
            A_total = A * (S + (1.0 - S) * c * mu_tilde)
            A_total = max(0.0, min(1.0, A_total))  # keep within [0,1]

            print(
                f"{base.name:25s}  A'={A_total:.4f}  "
                f"(A={A:.4f}, S={S:.3f}, μ̃={mu_tilde:.3f}, μ={mu:.3f}, "
                f"N={natoms}, R_eff={reff:.3f})"
            )

        except Exception as e:
            print(f"[warn] {cfile}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
