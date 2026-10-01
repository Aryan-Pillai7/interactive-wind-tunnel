"""D2Q9 lattice-Boltzmann ground-truth solver in numpy. (Owner: Aryan)

BGK collision; bounce-back on the obstacle and on the top/bottom walls
(rows 0 and NY-1); fixed-velocity inlet (u = U0, v = 0) on the left;
zero-gradient outlet on the right. nu = U0 * D / Re, tau = 3 * nu + 0.5.
"""


def solve(mask, re):
    """Run to steady state (or MAX_ITERS) for one obstacle.

    mask: bool [NY, NX], True inside the obstacle.
    re: Reynolds number in [RE_MIN, RE_MAX], based on U0 and D.

    Returns dict:
        u, v: float32 [NY, NX], lattice units, 0 inside the obstacle.
        p: float32 [NY, NX], gauge pressure coefficient (see contract), 0 inside.
        cd_total, cl_total: float, momentum-exchange force coefficients
            F / (0.5 * U0**2 * D) (pressure + viscous).
        iterations: int, steps taken.
        converged: bool, steady criterion met before MAX_ITERS.
        seconds: float, wall time of the solve.
    Raises ValueError if tau < TAU_MIN.
    """
    raise NotImplementedError
