# Electronegativity table
electronegativity = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    17: 3.16,  # Cl
    35: 2.96,  # Br
}

# Atoms considered as possible H-bond donors/acceptors
hb_atoms = [1, 7, 8, 9]

def hb_potential(atoms, charges=None):
    """
    atoms: list of tuples (x,y,z,atomic_number)
    charges: optional list of partial charges on atoms
    """
    potentials = []
    for i, (x,y,z,anum) in enumerate(atoms):
        if anum in hb_atoms:
            chi = electronegativity.get(anum,2.5)
            q = charges[i] if charges is not None else 0.0
            f = (chi/4.0)*(1 + abs(q)**0.5)  # normalize and include charge
            f = min(f, 1.0)
            potentials.append(f)
    if not potentials:
        return 0.0
    return max(potentials)  # or sum/potentials for average

# Example usage
atoms = [
    (0,0,0,8),  # O
    (0,0,1,1),  # H
    (1,0,0,6),  # C
]

charges = [-0.7, 0.4, 0.0]

f_hb = hb_potential(atoms, charges)
print(f"H-bonding potential factor: {f_hb:.3f}")

