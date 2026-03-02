import numpy as np
import logging
from sparse import COO




def check_g_degenerate(G):
    """
    Check if any rows of G are all zeros (degenerate constraints).
    """
    if not isinstance(G, COO):
        G = G.tocoo() if hasattr(G, "tocoo") else G

    NC, N = G.shape
    row_sums = np.zeros((NC,), dtype=float)
    for k in range(G.data.shape[0]):
        i = int(G.coords[0][k])
        v = float(G.data[k])
        row_sums[i] += abs(v)

    degenerate_rows = np.where(row_sums == 0.0)[0]
    n_degenerate = degenerate_rows.size

    if n_degenerate > 0:
        logging.warning(f"Found {n_degenerate} degenerate (all-zero) constraints in G.")
        return True, degenerate_rows.tolist()
    else:
        logging.info("No degenerate constraints found in G.")
        return False, []


def interval_for_pos(a_j, l_j, u_j, tiny, r_min_global, r_max_global):
    # requires l_j <= a_j * r <= u_j  => r in [l_j/a_j, u_j/a_j]
    if abs(a_j) <= tiny:
        # multiplicative update cannot move exact zero; if zero is already feasible any r is fine
        if (l_j <= 0.0 <= u_j):
            return (r_min_global, r_max_global)
        # otherwise nudge a_j to small nonzero in direction of midpoint so we can compute interval
        mid = (l_j + u_j) / 2.0
        sign = np.sign(mid) if mid != 0.0 else 1.0
        a0 = np.clip(sign * max(tiny, abs(l_j) if l_j != 0.0 else tiny), l_j, u_j)
        vals = np.array([l_j / a0, u_j / a0])
        return (float(np.min(vals)), float(np.max(vals)))
    else:
        vals = np.array([l_j / a_j, u_j / a_j])
        return (float(np.min(vals)), float(np.max(vals)))

def interval_for_neg(a_j, l_j, u_j, tiny, r_min_global, r_max_global):
    # requires l_j <= a_j / r <= u_j  => r in [a_j/u_j, a_j/l_j]
    # watch division by zero in l_j or u_j
    if abs(a_j) <= tiny:
        if (l_j <= 0.0 <= u_j):
            return (r_min_global, r_max_global)
        mid = (l_j + u_j) / 2.0
        sign = np.sign(mid) if mid != 0.0 else 1.0
        a0 = np.clip(sign * max(tiny, abs(l_j) if l_j != 0.0 else tiny), l_j, u_j)
        vals = []
        if abs(u_j) > tiny:
            vals.append(a0 / u_j)
        else:
            vals.append(np.sign(a0) * 1e12)
        if abs(l_j) > tiny:
            vals.append(a0 / l_j)
        else:
            vals.append(np.sign(a0) * 1e12)
        vals = np.array(vals)
        return (float(np.min(vals)), float(np.max(vals)))
    else:
        vals = []
        if abs(u_j) > tiny:
            vals.append(a_j / u_j)
        else:
            vals.append(np.sign(a_j) * 1e12)
        if abs(l_j) > tiny:
            vals.append(a_j / l_j)
        else:
            vals.append(np.sign(a_j) * 1e12)
        vals = np.array(vals)
        return (float(np.min(vals)), float(np.max(vals)))