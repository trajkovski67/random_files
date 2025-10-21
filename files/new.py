#!/usr/bin/env python3
"""
Compute attenuation factor using charge‑density, size, and
localization index based on partial charges, then
evaluate predicted energies vs experimental data.

Model:
    L_loc = 1 − (S_q / ln N),  S_q = −Σ p_i ln p_i,  p_i = |q_i| / Σ|q_i|
    A  = 1 / [1 + α (|q|/R_eff)^β]
    S  = 1 / [1 + exp(−k (N − N0))]
    A' = A * [ S + (1 − S) * c * L_loc ]

Energy test:
    error = A' * Gcorr_localb + ALPB − EXP
"""

import sys, math
from pathlib import Path
import numpy as np
# ─────────────────────────────────────────────
# Embedded dataset (solute : (EXP, ALPB, Gcorr_localb))
# ─────────────────────────────────────────────
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
    "chloroform_anion": (-54.1, -51.6, -6.7),
    "dichloroacetate": (-62.3, -56.2, -6.7),
    "2,2,2-trifluoroethanolate": (-77.5, -61.5, -16.2),
    "hexafluoropropanoate": (-65.5, -52.8, -13.3),
    "2-chlorophenolate": (-66.1, -54.8, -12.3),
    "4-chlorophenolate": (-66.0, -55.4, -10.7)
}

# ─────────────────────────────────────────────
# Input parsers
# ─────────────────────────────────────────────
def parse_cpcm(path: Path):
    """Read natoms, volume, and coordinates (Å)."""
    natoms = volume = None
    coords = []
    reading = False
    with open(path) as f:
        for line in f:
            if "Number of atoms" in line:
                try: natoms = int(line.strip().split()[0])
                except: pass
            if "Volume" in line:
                try: volume = float(line.strip().split()[0])
                except: pass
            if "# CARTESIAN COORDINATES" in line:
                reading = True; continue
            if reading:
                if line.strip().startswith("#") or not line.strip():
                    if coords: break
                    else: continue
                parts = line.split()
                try:
                    x, y, z = map(float, parts[:3])
                    coords.append((x, y, z))
                    if natoms and len(coords) >= natoms: break
                except: continue
    if natoms is None or volume is None:
        raise ValueError(f"{path}: failed to read natoms/volume")
    return natoms, volume, coords

def read_partial_charges(base: Path, n: int):
    fn = base / "alpb" / "charges"
    charges = [float(x) for x in fn.read_text().split()]
    if len(charges) != n:
        raise ValueError(f"{fn}: charges({len(charges)}) != natoms({n})")
    return charges

# ─────────────────────────────────────────────
# Physics functions
# ─────────────────────────────────────────────
def compute_reff(v_bohr3):
    bohr_to_ang = 0.529177
    v_ang3 = v_bohr3 * bohr_to_ang**3
    return (3*v_ang3/(4*math.pi))**(1/3)

def attenuation(alpha, beta, q, reff):
    return 1.0 / (1.0 + alpha*(abs(q)/reff)**beta)

def size_factor(n, n0=10.0, k=0.5):
    return 1.0 / (1.0 + math.exp(-k*(n-n0)))


def localization_entropy(charges, gamma=1.1):
    abs_q = np.abs(charges)
    w = abs_q**gamma
    p = w / w.sum()
    S = -np.sum(p * np.log(p))
    C = S / np.log(len(charges))
    return 1.0 - C



#def localization_entropy(charges):
#    """Localization index from charge entropy: 1 (localized) → 0 (delocalized)."""
#    abs_q = [abs(q) for q in charges]
#    tot = sum(abs_q)
#    if tot < 1e-12: 
#        return 1.0
#    p = [q/tot for q in abs_q]
#    S = -sum(pi*math.log(pi) for pi in p if pi>0)
#    C_entropy = S / math.log(len(charges))
#    return 1.0 - C_entropy

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 4:
        print("Usage: ./att_factor <alpha> <beta> <charge> [--n0 N0] [--k K] [--c C]")
        sys.exit(1)

    alpha, beta, charge = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    n0, k, c = 10.0, 0.5, 0.6
    if "--n0" in sys.argv: n0 = float(sys.argv[sys.argv.index("--n0")+1])
    if "--k"  in sys.argv: k  = float(sys.argv[sys.argv.index("--k")+1])
    if "--c"  in sys.argv: c  = float(sys.argv[sys.argv.index("--c")+1])

    root = Path(".")
    cpcm_files = list(root.rglob("molalign/inp.cpcm"))
    if not cpcm_files:
        sys.exit("No molalign/inp.cpcm files found.")

    errs = []
    for cfile in sorted(cpcm_files):
        base = cfile.parents[1]
        try:
            natoms, volume, coords = parse_cpcm(cfile)
            reff = compute_reff(volume)
            charges = read_partial_charges(base, natoms)
            if natoms==1: 
                L_loc = 0
            else:
                L_loc = localization_entropy(charges)

            A = attenuation(alpha, beta, charge, reff)
            S = size_factor(natoms, n0, k)
            A_total = A * (S + (1-S)*c*L_loc)
            A_total = max(0.0, min(1.0, A_total))

            name = base.name
            if name in DATA:
                EXP, ALPB, Gcorr = DATA[name]
                error = A_total*Gcorr + ALPB - EXP
                errs.append(abs(error))
                print(f"{name:25s}  A'={A_total:.4f}  "
                      f"(A={A:.4f}, S={S:.3f}, L_loc={L_loc:.3f}, "
                      f"N={natoms}, R_eff={reff:.3f}, error={error:+.3f})")
            else:
                print(f"{name:25s}  A'={A_total:.4f} (no EXP data)")

        except Exception as e:
            print(f"[warn] {cfile}: {e}", file=sys.stderr)

    if errs:
        mad = sum(errs)/len(errs)
        print(f"\nMean Abs Deviation (MAD) = {mad:.3f}")
        err = np.array(errs)
        mean_err = np.mean(err)
        std_err  = np.std(err)
        print(f"Variance =  {std_err:.3f}")
    
if __name__ == "__main__":
    main()
