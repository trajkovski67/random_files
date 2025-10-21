import os
import re

# Conversion factor: Hartree → kcal/mol
HARTREE_TO_KCAL = 627.509474

root_dir = "."  # current directory

for folder, subfolders, files in os.walk(root_dir):
    for file in files:
        if file == "out":  # look for ORCA output files
            out_path = os.path.join(folder, file)
            with open(out_path, "r", errors="ignore") as f:
                for line in f:
                    if "CPCM Dielectric" in line:
                        match = re.search(r"([-+]?\d*\.\d+|\d+)\s*Eh", line)
                        if match:
                            energy_au = float(match.group(1))
                            energy_kcal = energy_au * HARTREE_TO_KCAL
                            print(f"{out_path}: {energy_au:.8f} Eh = {energy_kcal:.3f} kcal/mol")
                        break

