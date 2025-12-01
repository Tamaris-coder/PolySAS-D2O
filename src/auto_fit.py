# src/auto_fit.py - Instant beautiful plot (runs in 5 seconds)
from sasmodels.core import load_model
from sasmodels.direct_model import call_kernel
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

model = load_model("core_shell_sphere")
kernel = model()

params = {
    'scale': 1.0,
    'background': 0.001,
    'sld_core': 1.8e-6,      # Hydrophobic core with doxorubicin
    'sld_shell': 5.9e-6,     # PEG shell
    'sld_solvent': 6.36e-6,  # D2O
    'radius_core': 60,
    'thickness_shell': 45,
}

q = np.logspace(-3, -0.6, 300)
I = call_kernel(kernel, params)

plt.figure(figsize=(10,7), facecolor='black')
plt.gca().set_facecolor('black')
plt.loglog(q, I, '-', lw=4, color='#00ffff', label='Doxorubicin-loaded PEG-PLA Micelle in D₂O')
plt.xlabel(r'q (Å⁻¹)', fontsize=14, color='white')
plt.ylabel(r'I(q) (cm⁻¹)', fontsize=14, color='white')
plt.title('PolySAS-D2O – Instant Core-Shell Model', color='#00ffff', fontsize=16)
plt.legend(fontsize=12, frameon=False)
plt.tick_params(colors='white')
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig('results/instant_micelle.png', dpi=300, facecolor='black')
plt.close()
print("Plot saved → results/instant_micelle.png")
