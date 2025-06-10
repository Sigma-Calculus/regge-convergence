#!/usr/bin/env python3
import mpmath as mp
mp.mp.dps = 50  # Set precision to 50 decimal places
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Norm and normalization using mpmath

def mp_norm(v):
    return mp.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def normalize(v):
    n = mp_norm(v)
    return (v[0]/n, v[1]/n, v[2]/n)

# Icosahedron base mesh
phi = (mp.mpf(1) + mp.sqrt(5)) / mp.mpf(2)
base_vertices = [
    normalize((mp.mpf(-1), phi, mp.mpf(0))),
    normalize((mp.mpf(1), phi, mp.mpf(0))),
    normalize((mp.mpf(-1), -phi, mp.mpf(0))),
    normalize((mp.mpf(1), -phi, mp.mpf(0))),
    normalize((mp.mpf(0), -mp.mpf(1), phi)),
    normalize((mp.mpf(0), mp.mpf(1), phi)),
    normalize((mp.mpf(0), -mp.mpf(1), -phi)),
    normalize((mp.mpf(0), mp.mpf(1), -phi)),
    normalize((phi, mp.mpf(0), -mp.mpf(1))),
    normalize((phi, mp.mpf(0), mp.mpf(1))),
    normalize((-phi, mp.mpf(0), -mp.mpf(1))),
    normalize((-phi, mp.mpf(0), mp.mpf(1))),
]
base_faces = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]

# Subdivision routine

def subdivide(vertices, faces):
    edge_mid = {}
    new_faces = []
    verts = vertices[:]

    def midpoint(i, j):
        key = tuple(sorted((i, j)))
        if key in edge_mid:
            return edge_mid[key]
        vi, vj = verts[i], verts[j]
        vm = normalize(((vi[0] + vj[0]) / 2,
                        (vi[1] + vj[1]) / 2,
                        (vi[2] + vj[2]) / 2))
        idx = len(verts)
        verts.append(vm)
        edge_mid[key] = idx
        return idx

    for tri in faces:
        i, j, k = tri
        a = midpoint(i, j)
        b = midpoint(j, k)
        c = midpoint(k, i)
        new_faces += [(i, a, c), (a, j, b), (c, b, k), (a, b, c)]

    return verts, new_faces

# Smooth C^∞ bump (Y20 spherical harmonic)

def perturb(v):
    x, y, z = v
    r = mp.mpf(1) + mp.mpf('0.3') * (3*z*z - mp.mpf(1)) / mp.mpf(2)
    return normalize((x*r, y*r, z*r))

# Custom clip for mpmath

def mp_clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

# Triangle area and angles

def triangle_area(va, vb, vc):
    x1, y1, z1 = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
    x2, y2, z2 = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
    cx = y1*z2 - z1*y2
    cy = z1*x2 - x1*z2
    cz = x1*y2 - y1*x2
    return mp.mpf('0.5') * mp_norm((cx, cy, cz))

def triangle_angles(va, vb, vc):
    def dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    a = (vb[0] - vc[0], vb[1] - vc[1], vb[2] - vc[2])
    b = (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2])
    c = (va[0] - vb[0], va[1] - vb[1], va[2] - vb[2])
    la, lb, lc = mp_norm(a), mp_norm(b), mp_norm(c)
    A = mp.acos(mp_clip(dot((vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]),
                             (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]))/(lc*lb), -1, 1))
    B = mp.acos(mp_clip(dot((va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]),
                             (vc[0] - vb[0], vc[1] - vb[1], vc[2] - vb[2]))/(la*lb), -1, 1))
    C = mp.acos(mp_clip(dot((va[0] - vc[0], va[1] - vc[1], va[2] - vc[2]),
                             (vb[0] - vc[0], vb[1] - vc[1], vb[2] - vc[2]))/(la*lc), -1, 1))
    return A, B, C

# Regge action implementing dual areas

def regge_action(vertices, faces):
    angle_sum = {i: mp.mpf('0') for i in range(len(vertices))}
    area_sum = {i: mp.mpf('0') for i in range(len(vertices))}
    for tri in faces:
        i, j, k = tri
        A, B, C = triangle_angles(vertices[i], vertices[j], vertices[k])
        angle_sum[i] += A
        angle_sum[j] += B
        angle_sum[k] += C
        area = triangle_area(vertices[i], vertices[j], vertices[k])
        for v in (i, j, k):
            area_sum[v] += area / mp.mpf(3)
    S = mp.mpf('0')
    for i in angle_sum:
        eps = mp.pi*mp.mpf('2') - angle_sum[i]
        S += eps * area_sum[i]
    return S * mp.mpf('0.5')

# Edge lengths calculation

def edge_lengths(vertices, faces):
    lengths = []
    for tri in faces:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            vi, vj = vertices[u], vertices[v]
            lengths.append(mp_norm((vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2])))
    return lengths

# Generate data

def compute_data(max_level=8):
    vertices, faces = base_vertices, base_faces
    data = []
    for level in range(max_level + 1):
        if level > 0:
            vertices, faces = subdivide(vertices, faces)
        verts_p = [perturb(v) for v in vertices]
        lengths = edge_lengths(verts_p, faces)
        h = mp.fsum(lengths) / mp.mpf(len(lengths))
        S = regge_action(verts_p, faces)
        data.append({'level': level, 'h': h, 'S': S})
    return data

# Main routine

def main():
    data = compute_data(max_level=9)
    levels = [d['level'] for d in data]
    h_list = [d['h'] for d in data]
    S_list = [d['S'] for d in data]
    err_list = [mp.fabs(S - S_list[-1]) for S in S_list]

    # Global fit range
    Lmin, Lmax = 3, 6
    idx = [i for i, l in enumerate(levels) if Lmin <= l <= Lmax]
    h_fit = [h_list[i] for i in idx]
    e_fit = [err_list[i] for i in idx]

    # Log-log linear fit
    xs = [mp.log(h) for h in h_fit]
    ys = [mp.log(e) for e in e_fit]
    mean_x = mp.fsum(xs) / len(xs)
    mean_y = mp.fsum(ys) / len(ys)
    nume = mp.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    deno = mp.fsum((x - mean_x)**2 for x in xs)
    p = nume / deno
    logC = mean_y - p * mean_x
    var_x = mp.fsum((x - mean_x)**2 for x in xs)
    var_y = mp.fsum((y - mean_y)**2 for y in ys)
    r2 = (nume**2) / (var_x * var_y)

    print(f"Globaler Fit: Levels {Lmin}-{Lmax}\n p = {p}\n R^2 = {r2}\n")

    # Lokale Exponenten paarweise
    print("Lokale Exponenten (i–i+1):")
    for i in range(len(err_list) - 1):
        if err_list[i + 1] != mp.mpf('0'):
            pi = mp.log(err_list[i] / err_list[i + 1]) / mp.log(h_list[i] / h_list[i + 1])
            print(f" Level {levels[i]}-{levels[i + 1]}: p = {pi}")

    # Richardson extrapolation
    i1, i2 = len(err_list) - 2, len(err_list) - 1
    hi, hi1 = h_list[i1], h_list[i2]
    ei, ei1 = err_list[i1], err_list[i2]
    E_ext = (hi1**2 * ei - hi**2 * ei1) / (hi1**2 - hi**2)
    print(f"\nRichardson-extrapolierter Fehler: {E_ext}\n")

    # Plot results
    hf = [float(h) for h in h_list]
    ef = [float(e) for e in err_list]
    fit_hf = [float(h) for h in h_fit]
    fit_ef = [float(e) for e in e_fit]

    plt.figure(figsize=(6, 5))
    plt.loglog(hf, ef, 'o', alpha=0.4, label='alle Level')
    plt.loglog(fit_hf, fit_ef, 'o', markerfacecolor='C1', markeredgecolor='k', label=f'Levels {Lmin}-{Lmax}')
    line_h = np.linspace(min(fit_hf), max(fit_hf), 100)
    plt.loglog(line_h, [float(mp.e**logC * (mp.mpf(lh)**p)) for lh in line_h], '--', label=f'p={float(p):.3f}')
    plt.xlabel('mean edge length h')
    plt.ylabel('|S - S_ref|')
    plt.title('Regge Konvergenz mit mpmath')
    plt.grid(which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    out = Path('regge_convergence.pdf')
    plt.savefig(out)
    print(f"Plot gespeichert nach {out}")

if __name__ == '__main__':
    main()
