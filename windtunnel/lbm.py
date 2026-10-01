"""D2Q9 lattice-Boltzmann ground-truth solver in numpy. (Owner: Aryan)

BGK collision; full-way bounce-back on the obstacle and on the top/bottom
walls (rows 0 and NY-1); fixed-velocity Zou/He inlet (u = U0, v = 0) on the
left; zero-gradient outlet on the right (unknown populations copied from the
previous column). nu = U0 * D / Re, tau = 3 * nu + 0.5.

Output conventions (see contract.py): u, v in lattice units; p is the gauge
pressure coefficient (rho - rho_ref) / 3 / (0.5 U0^2) with rho_ref the mean
density of the outlet column; u, v, p are 0 inside the obstacle. On the wall
rows u = v = 0 and p is copied from the adjacent fluid row.
"""

import time

import numpy as np

from windtunnel import contract as C
from windtunnel.geometry import obstacle_height

# Directions: rest, E, N, W, S, NE, NW, SW, SE.
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
EAST = np.array([1, 5, 8])   # cx = +1
WEST = np.array([3, 6, 7])   # cx = -1
MID = np.array([0, 2, 4])    # cx = 0
NORTH = np.array([2, 5, 6])  # cy = +1
SOUTH = np.array([4, 7, 8])  # cy = -1


def tau_for(re, d):
    """BGK relaxation time for Reynolds number re and characteristic length d."""
    return 3.0 * C.U0 * d / re + 0.5


def equilibrium(rho, ux, uy):
    out = np.empty((9,) + rho.shape, rho.dtype)
    base = 1.0 - 1.5 * (ux * ux + uy * uy)
    for q in range(9):
        if q == 0:
            out[0] = base
        else:
            cu = 3.0 * (CX[q] * ux + CY[q] * uy) if CX[q] and CY[q] else                 (3.0 * CX[q]) * ux if CX[q] else (3.0 * CY[q]) * uy
            out[q] = base + cu * (1.0 + 0.5 * cu)
        out[q] *= W[q] * rho
    return out


def macroscopic(f):
    rho = f.sum(axis=0)
    ux = np.tensordot(CX.astype(f.dtype), f, 1) / rho
    uy = np.tensordot(CY.astype(f.dtype), f, 1) / rho
    return rho, ux, uy


def _stream_index(ny, nx):
    """Flat gather index: f_new.ravel() = f_out.ravel()[index]."""
    jj, ii = np.mgrid[0:ny, 0:nx]
    n = ny * nx
    return np.concatenate([
        q * n + (((jj - CY[q]) % ny) * nx + (ii - CX[q]) % nx).ravel() for q in range(9)])


def _boundaries(f):
    """Apply outlet and inlet in place; return rho, ux, uy, feq."""
    f[WEST, :, -1] = f[WEST, :, -2]                            # zero-gradient outlet
    rho, ux, uy = macroscopic(f)
    ux[:, 0] = C.U0                                            # Zou/He velocity inlet
    uy[:, 0] = 0.0
    rho[:, 0] = (f[MID, :, 0].sum(axis=0) + 2.0 * f[WEST, :, 0].sum(axis=0)) / (1.0 - C.U0)
    feq = equilibrium(rho, ux, uy)
    f[EAST, :, 0] = feq[EAST, :, 0] + f[OPP[EAST], :, 0] - feq[OPP[EAST], :, 0]
    return rho, ux, uy, feq


def solve(mask, re, max_iters=C.MAX_ITERS, d=None, dtype=np.float64):
    """Run to steady state (or max_iters) for one obstacle.

    mask: bool [NY, NX], True inside the obstacle.
    re: Reynolds number in [RE_MIN, RE_MAX], based on U0 and D.
    d: characteristic length; defaults to obstacle_height(mask).

    Returns dict:
        u, v: float32 [NY, NX], lattice units, 0 inside the obstacle.
        p: float32 [NY, NX], gauge pressure coefficient (see contract), 0 inside.
        cd_total, cl_total: float, momentum-exchange force coefficients
            F / (0.5 * U0**2 * D) (pressure + viscous).
        iterations: int, steps taken.
        converged: bool, steady criterion met before max_iters.
        residual: float, last relative change of u over CONV_EVERY steps.
        seconds: float, wall time of the solve.
        tau, d: the relaxation time and characteristic length used.
    Raises ValueError if tau < TAU_MIN.
    """
    t0 = time.perf_counter()
    mask = np.asarray(mask, bool)
    d = float(obstacle_height(mask) if d is None else d)
    tau = tau_for(re, d)
    if tau < C.TAU_MIN:
        raise ValueError(f"tau={tau:.4f} < {C.TAU_MIN} (Re={re}, D={d})")
    omega = 1.0 / tau

    walls = np.zeros_like(mask)
    walls[0, :] = walls[-1, :] = True
    solid = mask | walls
    fluid = ~solid

    # Links from a fluid cell into an obstacle cell, per direction, for the
    # momentum-exchange force: population i at obstacle cell x came from x - c_i.
    links = [mask & np.roll(fluid, (CY[i], CX[i]), axis=(0, 1)) for i in range(9)]

    stream = _stream_index(*mask.shape)
    f = equilibrium(np.ones(mask.shape, dtype), (C.U0 * fluid).astype(dtype),
                    np.zeros(mask.shape, dtype))
    prev = None
    converged = False
    change = float("nan")
    it = 0
    for it in range(1, max_iters + 1):
        rho, ux, uy, feq = _boundaries(f)
        fout = f - omega * (f - feq)                           # BGK
        fout[:, solid] = f[:, solid][OPP]                      # bounce-back
        f = fout.ravel()[stream].reshape(f.shape)              # streaming

        if it % C.CONV_EVERY == 0:
            if not np.isfinite(ux).all():
                break
            u_now = ux[fluid]
            if prev is not None:
                change = np.linalg.norm(u_now - prev) / max(np.linalg.norm(u_now), 1e-30)
                if change < C.CONV_TOL:
                    converged = True
                    break
            prev = u_now.copy()

    fx = 2.0 * sum(CX[i] * f[i][links[i]].sum() for i in range(9))
    fy = 2.0 * sum(CY[i] * f[i][links[i]].sum() for i in range(9))
    scale = 0.5 * C.U0**2 * d

    rho, ux, uy, _ = _boundaries(f)
    rho_ref = rho[1:-1, -1].mean()
    p = (rho - rho_ref) / 3.0 / (0.5 * C.U0**2)
    p[0], p[-1] = p[1], p[-2]
    ux[walls] = uy[walls] = 0.0
    ux[mask] = uy[mask] = p[mask] = 0.0
    finite = bool(np.isfinite(ux).all() and np.isfinite(p).all())
    return {
        "u": ux.astype(np.float32), "v": uy.astype(np.float32), "p": p.astype(np.float32),
        "cd_total": float(fx / scale), "cl_total": float(fy / scale),
        "iterations": it, "converged": bool(converged and finite), "residual": float(change),
        "seconds": time.perf_counter() - t0, "tau": tau, "d": d,
    }
