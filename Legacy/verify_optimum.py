"""
Verifica si el z=0.572804 es realmente el óptimo global del modelo.
Evalúa z para:
  - Solución IPOPT (la "óptima local")
  - Naive 50/50 Buy & Hold   -> w(i,t) = 0.5 constantes, u=v=0
  - Naive 50/50 Rebalanceo   -> w(i,t) = 0.5 constantes, u,v computed
  - 100% SPX
  - 100% CMC200
"""
from pathlib import Path
from Legacy.basemodelGAMS import load_market_data, solve_portfolio

def z_of_policy(w_func, u_func, v_func, context, lam=0.10, cm=1.0):
    """Evalúa la función objetivo z para una política dada."""
    assets    = context["assets"]
    T_vals    = context["T_vals"]
    mu_mix    = context["mu_mix"]
    sigma_mix = context["sigma_mix"]
    c_base    = context["c_base"]
    theta     = {a: 1.0 for a in assets}

    z = 0.0
    for t in T_vals:
        # retorno esperado
        ret = sum(w_func(i, t) * mu_mix[i].loc[t] * theta[i] for i in assets)
        # varianza
        var = sum(w_func(i, t) * w_func(j, t) * sigma_mix[i][j].loc[t]
                  for i in assets for j in assets)
        # costos
        cost = sum(c_base[i] * cm * (u_func(i, t) + v_func(i, t))
                   for i in assets)
        z += ret - lam * var - cost
    return z


def build_naive_bh(context):
    """50/50 sin rebalanceo: u(i,t1)=0.0 (parte de w0), u=v=0 después."""
    assets  = context["assets"]
    T_vals  = context["T_vals"]
    w0      = context["w0"]
    r       = context["r"]

    # drift del B&H: w evoluciona con returns
    w = {}
    t0 = T_vals[0]
    for i in assets:
        w[i, t0] = w0[i]
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        denom  = sum(w[i, t_prev] * (1 + r[i].loc[t_prev]) for i in assets)
        for i in assets:
            w[i, t] = w[i, t_prev] * (1 + r[i].loc[t_prev]) / denom

    u = {(i, t): 0.0 for i in assets for t in T_vals}
    v = {(i, t): 0.0 for i in assets for t in T_vals}
    # anclaje inicial: w(t1)=w0, así que u=v=0 en t1 también
    return (lambda i, t: w[i, t],
            lambda i, t: u[i, t],
            lambda i, t: v[i, t])


def build_constant(w_target, context):
    """w(i,t) = w_target para todo t, rebalanceo explícito."""
    assets  = context["assets"]
    T_vals  = context["T_vals"]
    w0      = context["w0"]

    u, v = {}, {}
    t0 = T_vals[0]
    for i in assets:
        delta = w_target[i] - w0[i]
        u[i, t0] = max(delta, 0.0)
        v[i, t0] = max(-delta, 0.0)
    for idx in range(1, len(T_vals)):
        t = T_vals[idx]
        for i in assets:
            u[i, t] = 0.0  # w no cambia -> u=v=0
            v[i, t] = 0.0
    return (lambda i, t: w_target[i],
            lambda i, t: u[i, t],
            lambda i, t: v[i, t])


if __name__ == "__main__":
    base_path_str = str(Path(__file__).parent / "data")
    context = load_market_data(base_path_str)
    assets  = context["assets"]

    print("=" * 70)
    print("Evaluacion de z(ex-ante) para distintas politicas  (lambda=0.10)")
    print("=" * 70)

    # 1. IPOPT
    theta = {a: 1.0 for a in assets}
    z_ip, w_ip, u_ip, v_ip, _ = solve_portfolio(theta, context, lambda_riesgo=0.10)
    print(f"  IPOPT (lo que llama 'optimal_local')   z = {z_ip:.6f}")

    # 2. Naive 50/50 rebalanceo constante
    w_const, u_const, v_const = build_constant({"SPX": 0.5, "CMC200": 0.5}, context)
    z_rb = z_of_policy(w_const, u_const, v_const, context)
    print(f"  50/50 rebalanceo constante             z = {z_rb:.6f}")

    # 3. Naive 50/50 buy & hold (drift)
    w_bh, u_bh, v_bh = build_naive_bh(context)
    z_bh = z_of_policy(w_bh, u_bh, v_bh, context)
    print(f"  50/50 buy & hold (drift)               z = {z_bh:.6f}")

    # 4. 100% SPX
    w_spx, u_spx, v_spx = build_constant({"SPX": 1.0, "CMC200": 0.0}, context)
    z_spx = z_of_policy(w_spx, u_spx, v_spx, context)
    print(f"  100% SPX                               z = {z_spx:.6f}")

    # 5. 100% CMC200
    w_cmc, u_cmc, v_cmc = build_constant({"SPX": 0.0, "CMC200": 1.0}, context)
    z_cmc = z_of_policy(w_cmc, u_cmc, v_cmc, context)
    print(f"  100% CMC200                            z = {z_cmc:.6f}")

    print()
    print("  Si z(IPOPT) es el mayor -> IPOPT encontro el optimo global.")
    print("  Si alguna otra politica tiene mayor z -> IPOPT quedo atrapado.")
