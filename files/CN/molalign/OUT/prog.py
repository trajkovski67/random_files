#!/usr/bin/env python3
"""
corr_with_xtb_alpha_chargefix.py
--------------------------------
Compute implicit–explicit corrected solute–solvent interaction energies
for all JSON grid files in the current directory.  Uses xTB --alpha
to obtain the solvent's polarizability (in a.u.), converts it to Å³,
applies exponential damping to the implicit cross-term, and includes
solute charge = –1.
"""

import json, os, glob, math, subprocess, re, tempfile
import numpy as np
from math import pi, tanh
from collections import defaultdict

# ---------------- Constants ----------------
K_COULOMB = 332.06371
HARTREE_TO_KCAL = 627.509474
R_kcal = 0.0019872041
EPS_R = 78.5
T_DEFAULT = 298.15
AU_TO_ANG3 = 0.14818471      # 1 a.u. polarizability → Å³
SOLUTE_CHARGE = -1           # <== your solute charge

Z_TO_EL = {1:"H",6:"C",7:"N",8:"O",9:"F",15:"P",16:"S",17:"Cl",35:"Br",53:"I"}
BONDI = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,
         "P":1.80,"S":1.80,"Cl":1.75,"Br":1.85,"I":1.98}

OBC_ALPHA, OBC_BETA, OBC_GAMMA = 1.0, 0.8, 4.85
DESCREEN_SCALE = 0.8

# ---------------- Geometry helpers ----------------
def sphere_overlap(a,b,d):
    if d>=a+b: return 0.0
    if d<=abs(a-b): return (4/3)*pi*min(a,b)**3
    return (pi*(a+b-d)**2*(d**2+2*d*(a+b)-3*(a-b)**2))/(12*d)

def obc_born_radii(coords, elements):
    coords=np.asarray(coords,float)
    n=len(coords)
    r_intr=np.array([BONDI.get(el,1.7) for el in elements])
    V=(4/3)*pi*r_intr**3
    psi=np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            d=np.linalg.norm(coords[i]-coords[j])
            psi[i]+=sphere_overlap(r_intr[i],DESCREEN_SCALE*r_intr[j],d)/V[i]
    t=np.tanh(OBC_ALPHA*psi - OBC_BETA*psi**2 + OBC_GAMMA*psi**3)
    return r_intr/(1.0-t)

def f_gb(r,a_i,a_j):
    return math.sqrt(r*r + a_i*a_j*math.exp(-r*r/(4*a_i*a_j)))

# ---------------- XTB polarizability ----------------
def run_xtb_polarizability(xyz_coords, elements):
    """Run xTB --alpha and extract isotropic polarizability (Å³)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(elements)}\n\n")
            for el,(x,y,z) in zip(elements, xyz_coords):
                f.write(f"{el:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        cmd = ["xtb", xyz_path, "--alpha"]
        try:
            result = subprocess.run(cmd, cwd=tmpdir,
                                    capture_output=True, text=True, timeout=120)
            out = result.stdout + result.stderr
        except Exception as e:
            print(f"⚠️  XTB run failed: {e}")
            return None

        # Look for "Mol. α(0) /au"
        match = re.search(r"Mol\.\s*α\(0\)\s*/au\s*[:=]\s*([0-9.]+)", out)
        if match:
            alpha_au = float(match.group(1))
            alpha_ang3 = alpha_au * AU_TO_ANG3
            print(f"🧪 XTB polarizability = {alpha_au:.3f} a.u.  ({alpha_ang3:.3f} Å³)")
            return alpha_ang3
        else:
            print("⚠️  Could not find polarizability line in xTB output.")
            return None

# ---------------- GB cross-term ----------------
def gb_cross_energy(sol_xyz, sol_q, sol_alpha,
                    solv_xyz, solv_q, solv_alpha,
                    eps_r=EPS_R, polarizability=None,
                    eta=0.3, alpha0=1.0):
    scale=-(1.0-1.0/eps_r)*K_COULOMB
    e=0.0
    for i in range(len(sol_q)):
        for j in range(len(solv_q)):
            rij=np.linalg.norm(sol_xyz[i]-solv_xyz[j])
            fij=f_gb(rij,sol_alpha[i],solv_alpha[j])
            e+=sol_q[i]*solv_q[j]/fij
    e_cross=scale*e
    if polarizability is not None:
        damping=math.exp(-eta*polarizability/alpha0)
        e_cross*=damping
        #print(f"Applied damping exp(-{eta:.2f}*{polarizability:.2f}/{alpha0:.2f}) = {damping:.3f}")
    return e_cross

# ---------------- Boltzmann average ----------------
def boltzmann_weights(energies,T=T_DEFAULT):
    E=np.array(energies,float)
    Emin=np.min(E)
    beta=1.0/(R_kcal*T)
    w=np.exp(-beta*(E-Emin))
    w/=np.sum(w)
    Z=np.sum(np.exp(-beta*(E-Emin)))
    return w,Z,Emin

def boltzmann_average(values,energies,T=T_DEFAULT):
    w,Z,Emin=boltzmann_weights(energies,T)
    v=np.array(values,float)
    mean=float(np.sum(w*v))
    var=float(np.sum(w*(v-mean)**2))
    return mean, math.sqrt(var), {"Z":float(Z),"Emin_kcal":float(Emin)}

# ---------------- JSON processor ----------------
def process_json(json_in, solute_charges, solvent_charges=None, eta=0.3):
    with open(json_in) as f: data=json.load(f)
    solute=data["solute"]; solvent=data["solvent"]

    solute_xyz=np.array([[a["x"],a["y"],a["z"]] for a in solute["xyz"]])
    solvent_xyz=np.array([[a["x"],a["y"],a["z"]] for a in solvent["xyz"]])
    solute_el=[Z_TO_EL.get(int(a["element"]),str(a["element"])) for a in solute["xyz"]]
    solvent_el=[Z_TO_EL.get(int(a["element"]),str(a["element"])) for a in solvent["xyz"]]

    solute_q=np.asarray(solute_charges,float)
    if solvent_charges is None:
        solvent_q=np.array([-0.834,0.417,0.417]) if len(solvent_el)==3 else np.zeros(len(solvent_el))
    else:
        solvent_q=np.asarray(solvent_charges,float)

    solute_alpha=obc_born_radii(solute_xyz,solute_el)
    solvent_alpha=obc_born_radii(solvent_xyz,solvent_el)

    # xTB polarizability
    solvent_polar=run_xtb_polarizability(solvent_xyz, solvent_el)
    if solvent_polar is None: solvent_polar=3.0

    E_sol_ref=solute.get("energy_Eh",0.0)*HARTREE_TO_KCAL
    E_solv_ref=solvent.get("energy_Eh",0.0)*HARTREE_TO_KCAL
    n_sol=len(solute_el)
    energies=[]

    for gp_name,gp_data in data.items():
        if gp_name in ("solute","solvent"): continue
        if isinstance(gp_data,dict): pose_iter=gp_data.items()
        elif isinstance(gp_data,list): pose_iter=[(f"pose_{i}",p) for i,p in enumerate(gp_data)]
        else: continue
        for _,pose_data in pose_iter:
            while isinstance(pose_data,list) and len(pose_data)>0:
                pose_data=pose_data[0]
            if not isinstance(pose_data,dict): continue
            e_explicit=float(pose_data.get("energy_Eh",0.0))*HARTREE_TO_KCAL
            coords=np.array([[a["x"],a["y"],a["z"]] for a in pose_data.get("xyz",[])])
            sol_xyz=coords[:n_sol]; solv_xyz=coords[n_sol:]
            e_cross=gb_cross_energy(sol_xyz,solute_q,solute_alpha,
                                    solv_xyz,solvent_q,solvent_alpha,
                                    polarizability=solvent_polar,eta=eta)
            e_corr=e_explicit - e_cross
            e_int_corr=e_corr - (E_sol_ref + E_solv_ref)
            energies.append(e_int_corr)
    return energies

# ---------------- Main driver ----------------
def main():
    charge_file="charges"
    if not os.path.exists(charge_file):
        raise FileNotFoundError(f"Charge file '{charge_file}' not found.")
    with open(charge_file) as f:
        solute_charges=[float(x) for x in f if x.strip()]
   # print(f"✅ Loaded {len(solute_charges)} solute charges from '{charge_file}'")

    json_files=sorted(glob.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No .json files found.")
    print(f"📂 Found {len(json_files)} JSON files to process.\n")

    allE=[]
    for fn in json_files:
        try:
            E=process_json(fn, solute_charges)
            allE.extend(E)
            print(f"✅ {fn}: {len(E)} corrected energies collected.")
        except Exception as e:
            print(f"❌ {fn}: {e}")

    if not allE:
        raise RuntimeError("No energies collected.")
    mean,std,meta=boltzmann_average(allE,allE,T_DEFAULT)
    print("\n=== GLOBAL BOLTZMANN AVERAGE ===")
    print(f"T = {T_DEFAULT:.2f} K")
    print(f"Global ⟨E_int,corr⟩_B = {mean:.6f} ± {std:.6f} kcal/mol")
    print(f"(Z={meta['Z']:.6g}, Emin={meta['Emin_kcal']:.6f} kcal/mol)")
    print(f"Samples: {len(allE)}")
    with open("global_average.txt","w") as f:
        f.write(f"Global average: {mean:.6f} ± {std:.6f} kcal/mol\n")
        f.write(f"Samples: {len(allE)}\n")

if __name__=="__main__":
    main()

