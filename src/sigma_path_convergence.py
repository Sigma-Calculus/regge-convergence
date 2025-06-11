#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
#  sigma_path_convergence.py  –  rein σ-basierter Konvergenztest
#
#  Dieses Skript implementiert eine vollständig σ-konforme
#  Version des Konvergenztests, wie er in Appendix G beschrieben
#  wurde. Es basiert auf gerichteten Pfaden (σ-Pfaden) und misst
#  die additive Spannungskohärenz gemäß Theorem 6.1 (σ-Konvergenzsatz).
#
#  Zur Prüfung der Robustheit wurde σ.7 gezielt verletzt.
#  Das Mesh ist absichtlich nicht quasi-uniform – das ist Teil der Teststrategie.
#
#  Abschnittsbezug:
#    - σ-Pfade: Definition 2.1 (gerichtete Pfade)
#    - Spannung: Definition 2.2 (Tension functional)
#    - Halbierungsfehler: Definition 3.2 (σ–Fehlerfunktional)
#    - Konvergenzsatz: Theorem 6.1 (Quadratische Kohärenzreduktion)
#
#  * erzeugt icosahedrische Refinements einer radial gestörten
#    Sphäre  r(θ)=1+0.2·cos²θ
#  * misst pro Kante γ den Halbierungs-Fehler
#
#        Δγ = |Σ(γ) − Σ(γ₁) − Σ(γ₂)| / span(γ)
#
#    (γ₁, γ₂: zwei Teilpfade nach einmaliger Halbierung)
#  * prüft Δσ(h) ∝ h²  mittels Log-Log-Fit
#  * speichert Diagramm als PDF
# ────────────────────────────────────────────────────────────────
import math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------ 1. Hilfsfunktionen
def normalize(v):
    """Einheitsnorm im ℝ³"""
    return v / np.linalg.norm(v)

def perturb(v):
    """σ-Verformung: r = 1 + 0.2·z²"""
    x, y, z = v
    return normalize(v) * (1.0 + 0.2 * z * z)

def midpoint_sigma(vi, vj):
    """Pfadhalbierung im σ-Sinn (geometrisch mitteln + perturbieren)"""
    return perturb(normalize((vi + vj) / 2.0))

def sigma_edge(vi, vj):
    """Spannung einer Kante = euklidische Länge"""
    return np.linalg.norm(vj - vi)

# ------------------------------ 2. Icosaeder-Refinement
phi = (1 + 5 ** 0.5) / 2
ICO_VERTS = np.array([
    (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
    (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
    ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
], float)
ICO_VERTS = np.array([normalize(v) for v in ICO_VERTS])

ICO_FACES = np.array([
    (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
    (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
    (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
    (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)
], int)

def subdivide(vertices, faces):
    """Loop-Subdivison, anschließende Projektion auf Einheitskugel"""
    edge_mid, verts, new_faces = {}, vertices.tolist(), []

    def mid(i, j):
        key = tuple(sorted((i, j)))
        if key in edge_mid:
            return edge_mid[key]
        idx  = len(verts)
        edge_mid[key] = idx
        verts.append(normalize((np.array(verts[i]) + np.array(verts[j])) / 2.0))
        return idx

    for i, j, k in faces:
        a, b, c = mid(i, j), mid(j, k), mid(k, i)
        new_faces += [(i, a, c), (a, j, b), (c, b, k), (a, b, c)]
    return np.array(verts, float), np.array(new_faces, int)

def build_mesh(level):
    """liefert (Verformte Vertices, Kantenpaare)"""
    v, f = ICO_VERTS, ICO_FACES
    for _ in range(level):
        v, f = subdivide(v, f)
    v = np.array([perturb(p) for p in v])          # σ-Verformung
    edges = {(min(i, j), max(i, j)) for i, j, k in f
             for i, j in [(i, j), (j, k), (k, i)]}
    return v, list(edges)

# ------------------------------ 3. Δσ(h) via Halbierungsprinzip
def delta_sigma(level):
    """liefert (h,  max. Δσ) auf Level"""
    v, edges = build_mesh(level)
    spans, local_err = [], []
    for i, j in edges:
        vi, vj = v[i], v[j]
        span   = np.linalg.norm(vj - vi)
        vm     = midpoint_sigma(vi, vj)
        err    = abs(sigma_edge(vi, vj) -
                     sigma_edge(vi, vm) - sigma_edge(vm, vj)) / span
        spans.append(span)
        local_err.append(err)
    return np.mean(spans), max(local_err)

# ------------------------------ 4. Berechnung & Plot
rows = [(lvl,) + delta_sigma(lvl) for lvl in range(6)]
df   = pd.DataFrame(rows, columns=["level", "h", "Delta_sigma"])
print(df)

# Log-Log-Regression
p, logC = np.polyfit(np.log(df.h), np.log(df.Delta_sigma), 1)
C       = math.exp(logC)
print(f"Fitted slope  p ≈ {p:.3f}")

# Plot
plt.figure()
plt.loglog(df.h, df.Delta_sigma, "o", label="data")
h_line = np.linspace(df.h.max()*0.9, df.h.min()*1.1, 120)
plt.loglog(h_line, C*h_line**p, "--", label=f"fit: p={p:.2f}")
plt.xlabel(r"mean span $h$")
plt.ylabel(r"$\Delta_\sigma(h)$")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
out = Path("sigma_path_convergence.pdf")
plt.savefig(out)
print("Plot saved to", out.resolve())
