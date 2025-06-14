import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Parameter und Zufalls-Setup
# ------------------------------------------------------------------

"""
Größeres N ⇒ Mehr Segmente ⇒ jedes Δσᵢ wird kleiner ⇒ 
kartenartige Kanten werden feiner und das Gesamtsignal wirkt glatter.
"""
N         = 200                         # Zahl der Pfadpunkte

rng       = np.random.default_rng(42)   # Reproduzierbare Zufallszahlen
z_target  = 1.0                         # gewünschte Rotverschiebung
a_emit    = 1 / (1 + z_target)          # Skalenfaktor beim Sender (0.5)

# ------------------------------------------------------------------
# 2. ΛCDM-Zweig (glatter Pfad)
# ------------------------------------------------------------------
Omega_m, Omega_L, H0 = 0.31, 0.69, 70.0        # Planck-ähnliche Werte
a_grid = np.linspace(a_emit, 1.0, N)           # Skalenfaktor a(t)

H_a    = H0 * np.sqrt(Omega_m / a_grid**3 + Omega_L)
dt_da  = 1 / (a_grid * H_a)                    # dt/da

# numerische Integration (Trapezregel)
t_grid = np.zeros_like(a_grid)
for i in range(1, N):
    h = a_grid[i] - a_grid[i-1]
    t_grid[i] = t_grid[i-1] + 0.5 * (dt_da[i] + dt_da[i-1]) * h

s = (t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0])   # normierter Pfadparameter

r_cdm      = a_grid.copy()            # r(s) = a(t)
sigma_cdm  = r_cdm[0] / r_cdm         # Transportgesetz: σ = r(0)/r(s)
prod_cdm   = r_cdm * sigma_cdm
z_sigma_cdm = sigma_cdm[0] / sigma_cdm[-1] - 1        # dokumentkonform

# ------------------------------------------------------------------
# 3. σ-Diskret-Zweig (gezackter Pfad)
# ------------------------------------------------------------------
inc_raw   = rng.uniform(0.5, 1.5, size=N-1)            # positive Inkremente
base_step = (1.0 - a_emit) / (N - 1)
inc       = inc_raw * base_step
inc      *= (1.0 - a_emit) / np.sum(inc)               # Reskalieren auf Δa

r_sigma = np.empty(N)
r_sigma[0] = a_emit
for i in range(1, N):
    r_sigma[i] = r_sigma[i-1] + inc[i-1]
r_sigma[-1] = 1.0

sigma_sigma  = r_sigma[0] / r_sigma
prod_sigma   = r_sigma * sigma_sigma
z_sigma_alt  = sigma_sigma[0] / sigma_sigma[-1] - 1    # sollte 1 sein

# ------------------------------------------------------------------
# 4. Plot beider Pfade
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))

# r(s)
plt.plot(s, r_cdm,       label=r"$r_{\Lambda\text{CDM}}(s)$",            lw=2)
plt.plot(s, r_sigma,     label=r"$r_{\sigma\text{-Pfad}}(s)$",           lw=2, ls="--")

# σ(s)
plt.plot(s, sigma_cdm,   label=r"$\sigma_{\Lambda\text{CDM}}(s)$",       lw=1.2)
plt.plot(s, sigma_sigma, label=r"$\sigma_{\sigma\text{-Pfad}}(s)$",      lw=1.2, ls="--")

# konstante Produkte
plt.plot(s, prod_cdm,   label=r"$r\,\sigma$ const. (CDM)",      color="red",   lw=1)
plt.plot(s, prod_sigma, label=r"$r\,\sigma$ const. (σ-Pfad)",   color="magenta", lw=1)

plt.xlabel("Pfadparameter $s$")
plt.ylabel("Wert")
plt.title("Vergleich: glatter ΛCDM-Pfad vs. gezackter σ-Pfad")
plt.legend(loc="best", fontsize=8)
plt.grid(True)
plt.tight_layout()

plt.savefig("cdm_vs_sigma_path.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
# 5. Zusammenfassung ausgeben
# ------------------------------------------------------------------
print("Dokument-konforme Rotverschiebungen:")
print(f"  ΛCDM-Zweig   z_σ = {z_sigma_cdm:.3f}")
print(f"  σ-Zweig      z_σ = {z_sigma_alt:.3f}")
print()
print("Konstanzprüfung:")
print(f"  max|rσ_CDM − const|   = {np.max(np.abs(prod_cdm   - prod_cdm[0])):.2e}")
print(f"  max|rσ_σ  − const|    = {np.max(np.abs(prod_sigma - prod_sigma[0])):.2e}")
