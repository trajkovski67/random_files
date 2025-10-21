#!/usr/bin/env python3
"""
Compute attenuation factor using charge-density, size, localization,
and polarizability density (α(0)/V) from xTB, then evaluate predicted
energies vs experimental data.

Model:
    L_loc = 1 − (S_q / ln N),  S_q = −Σ p_i ln p_i,  p_i = |q_i| / Σ|q_i|
    A  = 1 / [1 + α (|q|/R_eff)^β]
    S  = 1 / [1 + exp(−k (N − N0))]
    A' = A * [ S + (1 − S) * c * L_loc ]
    α̃_V = α(0)/V
    P = 1 / [1 + (α̃_V / α0)^γ]
    A'' = A' * P
    error = A'' * Gcorr_localb + ALPB − EXP
"""

import sys, math, re, subprocess, tempfile, random
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
    "water": (-104.7, -91.6, -19.2),
    "sulfide": (-72.1, -64.6, -12.2),
    "thiophenol": (-63.4, -50.8, -12.3)
}

# ─────────────────────────────────────────────
# Parsing functions
# ─────────────────────────────────────────────
def parse_cpcm(path: Path):
    """Read natoms, volume (bohr³), coordinates, and element symbols."""
    natoms = volume = None
    coords, elems = [], []
    reading = False
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if "Number of atoms" in line:
                try:
                    natoms = int(line.strip().split()[0])
                except:
                    pass
            if "Volume" in line:
                floats = [w for w in line.replace(":", " ").split()
                          if re.match(r"^[+-]?\d+(\.\d+)?$", w)]
                if floats:
                    volume = float(floats[-1])
            if "# CARTESIAN COORDINATES" in line:
                reading = True
                continue
            if reading:
                if line.strip().startswith("#") or not line.strip():
                    if coords:
                        break
                    else:
                        continue
                parts = line.split()
                try:
                    x, y, z = map(float, parts[:3])
                    sym = None
                    if len(parts) >= 4 and parts[3].isalpha() and len(parts[3]) <= 2:
                        sym = parts[3].title()
                    elif parts[-1].isalpha() and len(parts[-1]) <= 2:
                        sym = parts[-1].title()
                    coords.append((x, y, z))
                    elems.append(sym or "C")
                    if natoms and len(coords) >= natoms:
                        break
                except:
                    continue
    if natoms is None or volume is None:
        raise ValueError(f"{path}: failed to read natoms/volume")
    return natoms, volume, coords, elems

def read_partial_charges(base: Path, n: int):
    fn = base / "alpb" / "charges"
    charges = [float(x) for x in fn.read_text().split()]
    if len(charges) != n:
        raise ValueError(f"{fn}: charges({len(charges)}) != natoms({n})")
    return charges

# ─────────────────────────────────────────────
# Physics utilities
# ─────────────────────────────────────────────
def compute_reff_from_volume(v_bohr3):
    bohr_to_ang = 0.529177
    v_ang3 = v_bohr3 * bohr_to_ang**3
    return (3*v_ang3/(4*math.pi))**(1/3)

def attenuation(alpha, beta, q, reff):
    return 1.0 / (1.0 + alpha*(abs(q)/reff)**beta)

def size_factor(n, n0=10.0, k=0.5):
    return 1.0 / (1.0 + math.exp(-k*(n-n0)))

def localization_entropy(charges):
    abs_q = [abs(q) for q in charges]
    tot = sum(abs_q)
    if tot < 1e-12: return 1.0
    p = [q/tot for q in abs_q]
    S = -sum(pi*math.log(pi) for pi in p if pi>0)
    return 1.0 - S / math.log(len(charges))

# ─────────────────────────────────────────────
# xTB polarizability helpers
# ─────────────────────────────────────────────
XTB_POL_RE = re.compile(r"Mol\.\s*α\(0\)\s*/au\s*:\s*([0-9]*\.?[0-9]+)")

def write_xyz(tmpdir: Path, name: str, elems, coords):
    xyz = tmpdir / f"{name}.xyz"
    with open(xyz, "w") as f:
        f.write(f"{len(coords)}\n{name}\n")
        for e, (x, y, z) in zip(elems, coords):
            f.write(f"{e:2s}  {x: .8f} {y: .8f} {z: .8f}\n")
    return xyz

def run_xtb_polarizability(xyz_path: Path, charge: int, gfn: int = 2, cwd: Path | None = None):
    cmd = ["xtb", str(xyz_path), "--gfn", str(gfn), "--chrg", str(charge)]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    m = XTB_POL_RE.search(out)
    if not m:
        propfile = (cwd or Path.cwd()) / "properties.out"
        if propfile.exists():
            txt = propfile.read_text(errors="ignore")
            m = XTB_POL_RE.search(txt)
    if not m:
        raise RuntimeError("xTB polarizability not found.")
    return float(m.group(1))

def polarizability_factor(alpha0_au, volume_bohr3, alpha0_param, gamma):
    """Polarizability damping using α(0)/V."""
    alpha_tilde = alpha0_au / max(1e-12, volume_bohr3)
    return 1.0 / (1.0 + (alpha_tilde / max(1e-12, alpha0_param))**gamma)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 4:
        print("Usage: ./att_factor <alpha> <beta> <charge> "
              "[--n0 N0] [--k K] [--c C] "
              "[--alpha0 A0] [--gamma G] "
              "[--no-xtb] [--cache]")
        sys.exit(1)

    alpha, beta, charge = map(float, sys.argv[1:4])
    n0, k, c = 10.0, 0.5, 0.0   # c fixed to 0 as requested
    alpha0_param, gamma = 0.8, 2.0
    use_xtb, cache = True, False
    gfn = 2

    args = sys.argv
    if "--n0" in args: n0 = float(args[args.index("--n0")+1])
    if "--k"  in args: k  = float(args[args.index("--k")+1])
    if "--alpha0" in args: alpha0_param = float(args[args.index("--alpha0")+1])
    if "--gamma"  in args: gamma = float(args[args.index("--gamma")+1])
    if "--gfn"    in args: gfn = int(args[args.index("--gfn")+1])
    if "--no-xtb" in args: use_xtb = False
    if "--cache"  in args: cache = True

    cpcm_files = list(Path(".").rglob("molalign/inp.cpcm"))
    if not cpcm_files:
        sys.exit("No molalign/inp.cpcm files found.")

    errs = []
    for cfile in sorted(cpcm_files):
        base = cfile.parents[1]
        try:
            natoms, volume, coords, elems = parse_cpcm(cfile)
            reff = compute_reff_from_volume(volume)
            charges = read_partial_charges(base, natoms)
            L_loc = 0.0 if natoms == 1 else localization_entropy(charges)

            A = attenuation(alpha, beta, charge, reff)
            S = size_factor(natoms, n0, k)
            A_prime = A * S
            A_prime = max(0.0, min(1.0, A_prime))

            # xTB polarizability
            P, alpha0_au = 1.0, None
            cache_file = base / "alpb" / "alpha0_xtb.dat"
            if use_xtb:
                if cache and cache_file.exists():
                    try:
                        alpha0_au = float(cache_file.read_text().strip())
                    except:
                        alpha0_au = None
                if alpha0_au is None:
                    with tempfile.TemporaryDirectory() as td:
                        tdir = Path(td)
                        xyz = write_xyz(tdir, base.name, elems, coords)
                        (tdir / ".CHRG").write_text(str(int(round(charge))))
                        alpha0_au = run_xtb_polarizability(xyz, int(round(charge)), gfn=gfn, cwd=tdir)
                    if cache:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(f"{alpha0_au:.6f}\n")
                P = polarizability_factor(alpha0_au, volume, alpha0_param, gamma)

            A_total = max(0.0, min(1.0, A_prime * P))

            name = base.name
            if name in DATA:
                EXP, ALPB, Gcorr = DATA[name]
                error = A_total * Gcorr + ALPB - EXP
                errs.append(abs(error))
                pol_txt = f", α0={alpha0_au:.3f} au, P={P:.3f}" if (use_xtb and alpha0_au is not None) else ""
                print(f"{name:25s}  A''={A_total:.4f}  "
                      f"(A'={A_prime:.4f}, A={A:.4f}, S={S:.3f}, "
                      f"N={natoms}, V={volume:.3f}{pol_txt}, error={error:+.3f})")
            else:
                pol_txt = f", α0={alpha0_au:.3f} au, P={P:.3f}" if (use_xtb and alpha0_au is not None) else ""
                print(f"{name:25s}  A''={A_total:.4f} (no EXP data, N={natoms}, V={volume:.3f}{pol_txt})")

        except Exception as e:
            print(f"[warn] {cfile}: {e}", file=sys.stderr)

    if errs:
        mad = sum(errs) / len(errs)
        err = np.array(errs)
        std_err = np.std(err)
        print(f"\nMean Abs Deviation (MAD) = {mad:.3f}")
        print(f"Variance =  {std_err:.3f}")

if __name__ == "__main__":
    main()

