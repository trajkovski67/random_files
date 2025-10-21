#!/usr/bin/env python3
"""
Compute attenuation factor with charge‑density, size, and
largest‑bond (pairwise) dipole correction.

For each pair of atoms i,j:
    μ_ij = |q_i − q_j| * d_ij          (e·Å)
Then μ_max = max(μ_ij)
Normalized dipole: μ̃ = μ_max / R_eff

Model:
    A  = 1 / [1 + α (|q|/R_eff)^β]
    S  = 1 / [1 + exp(−k (N − N0))]
    A' = A * [ S + (1 − S) * c * μ̃ ]

Expected structure:
    molecule/
      ├── molalign/inp.cpcm
      └── alpb/charges

Usage:
    ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K] [--c C]
"""

import sys, math
from pathlib import Path
from itertools import combinations

DATA = {
    "acetylene": (-76.5, -75.8, -21.2),
    "acetonitrile_anion": (-66.6, -67.6, -14.9),
    "cyanamide": (-72.2, -73.1, -16.3),
    "aniline": (-62.9, -55.3, -14.7),
    "diphenylamine_anion": (-54.6, -44.7, -11.3),
    "CN": (-70.2, -78.8, -17.6),
    "formicacid": (-76.2, -68.9, -14.3),
    "formate": (-77.6, -67.2, -13.7),
    "propanoate": (-76.2, -66.5, -13.9),
    "hexanoate": (-74.6, -66.0, -14.4),
    "acrylate": (-74.0, -66.2, -14.3),
    "pyruvate": (-68.5, -63.8, -12.9),
    "benzoate": (-71.2, -63.9, -13.2),
    "methanolate": (-95.0, -75.4, -20.0),
    "ethanolate": (-90.7, -72.7, -20.0),
    "1-propanoate": (-88.3, -72.0, -19.8),
    "isopropanoate": (-86.3, -69.3, -19.1),
    "2-butanol": (-84.2, -67.9, -20.0),
    "t-butanolate": (-82.3, -65.9, -20.2),
    "allylalcohol": (-86.6, -70.1, -18.9),
    "benzylalcoholate": (-85.1, -66.8, -17.9),
    "2-methoxyethanol": (-89.4, -72.4, -18.5),
    "phenol": (-71.9, -57.1, -14.0),
    "2-methylphenolate": (-70.2, -56.2, -13.7),
    "3-methylphenolate": (-71.1, -57.8, -13.9),
    "4-methylphenolate": (-72.0, -56.8, -13.9),
    "1,2-ethanediolate": (-85.3, -65.5, -15.4),
    "3-hydroxyphenolate": (-73.8, -59.3, -13.4),
    "4-hydroxyphenolate": (-77.6, -59.8, -14.0),
    "methylhydroxydroperoxide": (-93.2, -81.8, -19.5),
    "ethylhydroxyperoxide": (-89.2, -80.9, -19.8),
    "acetaldehyde": (-76.5, -67.1, -15.7),
    "acetone_enolate": (-76.2, -64.2, -16.0),
    "3-pentanone-enolate": (-73.7, -61.1, -15.7),
    "water": (-104.7, -91.6, -19.2),
    "hydrogenperoxide_anion": (-97.3, -88.1, -20.2),
    "hydroperoxylradical": (-83.3, -78.5, -15.6),
    "2-nitrophenolate": (-60.1, -55.7, -11.9),
    "3-nitrophenol": (-61.9, -50.4, -11.8),
    "4-nitrophenol": (-57.8, -49.9, -9.1),
    "nitromethane_anion": (-76.5, -66.4, -14.2),
    "4-nitroaniline": (-57.4, -52.4, -10.2),
    "acetamide_anion": (-80.2, -68.2, -14.2),
    "methanethilate": (-73.8, -61.6, -11.8),
    "ethanethiol": (-71.8, -60.2, -12.1),
    "1-propanethiolate": (-71.8, -60.3, -12.0),
    "thiophenol": (-63.4, -50.8, -12.3),
    "sulfide": (-72.1, -64.6, -12.2),
    "DMSO_anion": (-67.7, -66.8, -13.7),
    "fluoride": (-104.4, -107.2, -20.0),
    "chloride": (-74.5, -71.4, -11.3),
    "bromide": (-68.3, -70.13, -9.1),
    "trifluoroacetate": (-59.3, -54.5, -7.0),
    "chloroacetate": (-69.7, -62.4, -12.1),
    "chloroform_anion":(-54.1, -51.6, -6.7),
    "dichloroacetate": (-62.3, -56.2, -6.7),
    "2,2,2-trifluoroethanolate": (-77.5, -61.5, -16.2),
    "hexafluoropropanoate": (-65.5, -52.8, -13.3),
    "2-chlorophenolate": (-66.1, -54.8, -12.3),
    "4-chlorophenolate": (-66.0, -55.4, -10.7)
}
# ─────────────────────────────────────────────
# Parse input files
# ─────────────────────────────────────────────
def parse_cpcm(path: Path):
    """Extract (natoms, volume_bohr3, coords) from inp.cpcm."""
    natoms, volume = None, None
    coords = []
    reading_coords = False
    with open(path) as f:
        for line in f:
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
        raise ValueError(f"Coordinate count {len(coords)} != {natoms}")
    return natoms, volume, coords


def read_partial_charges(base_dir: Path, natoms: int):
    """Read partial atomic charges (space/newline separated)."""
    charges_path = base_dir / "alpb" / "charges"
    if not charges_path.exists():
        raise FileNotFoundError(f"No charges file found: {charges_path}")
    text = charges_path.read_text().strip()
    charges = [float(x) for x in text.split()]
    if len(charges) != natoms:
        raise ValueError(f"Charge count ({len(charges)}) != natoms ({natoms}) in {charges_path}")
    return charges


# ─────────────────────────────────────────────
# Core computations
# ─────────────────────────────────────────────
def compute_reff(volume_bohr3: float) -> float:
    """Convert volume (bohr³) to effective radius (Å)."""
    bohr_to_ang = 0.529177
    v_ang3 = volume_bohr3 * (bohr_to_ang ** 3)
    return (3.0 * v_ang3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def compute_max_pair_dipole(charges, coords):
    """Compute the largest pairwise dipole magnitude in Å·e."""
    bohr_to_ang = 0.529177
    n = len(coords)
    mu_max = 0.0
    for i, j in combinations(range(n), 2):
        qdiff = abs(charges[i] - charges[j])
        (x1, y1, z1) = coords[i]
        (x2, y2, z2) = coords[j]
        dx = (x1 - x2) * bohr_to_ang
        dy = (y1 - y2) * bohr_to_ang
        dz = (z1 - z2) * bohr_to_ang
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        mu_ij = qdiff * d
        if mu_ij > mu_max:
            mu_max = mu_ij
    return mu_max


def attenuation(alpha, beta, q, reff):
    """Base charge‑density term (0–1)."""
    return 1.0 / (1.0 + alpha * (abs(q) / reff) ** beta)


def size_factor(natoms, n0=10.0, k=0.5):
    """Sigmoid‑like size factor (0–1)."""
    return 1.0 / (1.0 + math.exp(-k * (natoms - n0)))


# ─────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 4:
        print("Usage: ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K] [--c C]")
        sys.exit(1)

    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    charge = float(sys.argv[3])
    n0 = 10.0
    k = 0.5
    c = 0.6
    r=0
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
    l=[]
    for cfile in sorted(cpcm_files):
        base = cfile.parents[1]
        try:
            natoms, volume, coords = parse_cpcm(cfile)
            reff = compute_reff(volume)
            charges = read_partial_charges(base, natoms)
            mu_max = compute_max_pair_dipole(charges, coords)
            mu_tilde = mu_max / reff

            A = attenuation(alpha, beta, charge, reff)
            S = size_factor(natoms, n0, k)

            A_total = A * (S + (1.0 - S) * c * mu_tilde)
            A_total = max(0.0, min(1.0, A_total))

                        # --- New energy term ---
            base_name = base.name
            if base_name in DATA:
                d0, d1, d2 = DATA[base_name]
                energy_term = A_total * d2 + d1 - d0
            else:
                energy_term = 0.0  # or 0.0 if you prefer
            l.append(abs(energy_term))
            r = r + 1
            print(
                f"{base_name:25s}  A'={A_total:.4f}  "
                f"(A={A:.4f}, S={S:.3f}, μ={mu_tilde:.3f}, μ_max={mu_max:.3f}, "
                f"N={natoms}, R_eff={reff:.3f}, error={energy_term:.3f})"
            )


        except Exception as e:
            print(f"[warn] {cfile}: {e}", file=sys.stderr)
    print('MAD =')
    mad=sum(l)
    print(mad/r)

if __name__ == "__main__":
    main()
