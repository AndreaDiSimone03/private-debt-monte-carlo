"""
Monte Carlo simulation of a hypothetical real estate private debt portfolio
---------------------------------------------------------------------------
ALL DATA IN THIS PROJECT IS HYPOTHETICAL. The loans are generated at random from the ranges
below and do not describe any real loan, borrower or company.

What the script does:
  1. Builds a portfolio of bullet loans secured on real estate (CHF).
  2. Computes the contractual cash flows, the portfolio IRR and the Macaulay duration
     (no defaults).
  3. Runs a Monte Carlo simulation with correlated defaults (one-factor model) and
     recoveries that depend on the property value at default.
  4. Reports the distribution of the portfolio IRR and of credit losses, and shows how
     default correlation changes the tail. The expected number of defaults is the same for
     any correlation, but losses are not: with correlation, defaults cluster in bad years
     when property values also fall.

Usage:
    python private_debt_mc.py

Outputs (in ./output):
    loan_tape.csv         the hypothetical loans
    mc_results.csv        summary statistics for each correlation level
    private_debt_mc.png   dashboard of the simulation results
"""

import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (all assumptions are illustrative, not calibrated to real data)
# --------------------------------------------------------------------------
SEED = 7
N_LOANS = 25
N_SIM = 20_000
HORIZON = 5  # longest maturity in years

PRINCIPAL_RANGE = (2.0, 10.0)  # CHF million
LTV_RANGE = (0.50, 0.75)  # loan to value at origination
MATURITIES = (2, 3, 4, 5)  # years, bullet repayment
COUPON_BASE, COUPON_SLOPE = 0.040, 0.08  # coupon = base + slope * (LTV - 0.50)
PD_BASE, PD_SLOPE = 0.008, 0.040  # annual PD = base + slope * (LTV - 0.50)

FIRE_SALE_COST = 0.08  # legal and sale costs, share of collateral value
PRICE_MEAN, PRICE_VOL = 0.0, 0.25  # property value change at default (stressed, illustrative)
PRICE_CORR = 0.8  # correlation of property values with the systematic factor

RHO_BASE = 0.15  # asset correlation used in the main run
RHO_SENSITIVITY = (0.0, 0.15, 0.30)

OUT_DIR = Path("output")


# --------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------
def make_tape(rng) -> pd.DataFrame:
    ltv = rng.uniform(*LTV_RANGE, N_LOANS)
    tape = pd.DataFrame({
        "loan": [f"Loan {i + 1:02d}" for i in range(N_LOANS)],
        "principal_chf_m": rng.uniform(*PRINCIPAL_RANGE, N_LOANS).round(1),
        "ltv": ltv.round(3),
        "coupon": (COUPON_BASE + COUPON_SLOPE * (ltv - 0.50)).round(4),
        "maturity_years": rng.choice(MATURITIES, N_LOANS),
        "annual_pd": (PD_BASE + PD_SLOPE * (ltv - 0.50)).round(4),
    })
    return tape


def contractual_cash_flows(tape: pd.DataFrame) -> np.ndarray:
    """Cash flow at years 1..HORIZON if nothing defaults (interest yearly, bullet at maturity)."""
    cf = np.zeros(HORIZON)
    for _, loan in tape.iterrows():
        for t in range(1, int(loan.maturity_years) + 1):
            cf[t - 1] += loan.principal_chf_m * loan.coupon
        cf[int(loan.maturity_years) - 1] += loan.principal_chf_m
    return cf


# --------------------------------------------------------------------------
# IRR and duration
# --------------------------------------------------------------------------
def irr(cash_flows: np.ndarray, invested: float) -> np.ndarray:
    """IRR of many cash flow rows at once (times 1..K), by bisection. Shape (S, K) -> (S,)."""
    cf = np.atleast_2d(cash_flows)
    times = np.arange(1, cf.shape[1] + 1)
    lo = np.full(cf.shape[0], -0.99)
    hi = np.full(cf.shape[0], 0.50)
    for _ in range(60):
        mid = (lo + hi) / 2
        npv = (cf / (1 + mid)[:, None] ** times).sum(axis=1) - invested
        lo = np.where(npv > 0, mid, lo)  # npv falls as the rate rises
        hi = np.where(npv > 0, hi, mid)
    return (lo + hi) / 2


def macaulay_duration(cf: np.ndarray, y: float) -> float:
    times = np.arange(1, len(cf) + 1)
    pv = cf / (1 + y) ** times
    return float((times * pv).sum() / pv.sum())


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def simulate(tape: pd.DataFrame, rho: float, rng) -> dict:
    """
    One-factor default model. In each year a loan defaults if
        sqrt(rho) * Z + sqrt(1 - rho) * e  <  inverse_normal(PD)
    with Z the systematic factor (shared by all loans) and e an idiosyncratic shock.
    Recovery = collateral value after a property price shock (linked to Z), minus costs,
    capped at the principal. It is received one year after default.
    """
    S, L = N_SIM, len(tape)
    principal = tape.principal_chf_m.to_numpy()
    coupon = tape.coupon.to_numpy()
    maturity = tape.maturity_years.to_numpy()
    ltv = tape.ltv.to_numpy()
    threshold = np.array([statistics.NormalDist().inv_cdf(p) for p in tape.annual_pd])

    cf = np.zeros((S, HORIZON + 1))  # a recovery after a year-5 default arrives in year 6
    alive = np.ones((S, L), dtype=bool)
    defaulted = np.zeros((S, L), dtype=bool)
    loss = np.zeros((S, L))
    recovered = np.zeros((S, L))

    for t in range(1, HORIZON + 1):
        z = rng.standard_normal((S, 1))
        eps = rng.standard_normal((S, L))
        eta = rng.standard_normal((S, L))
        asset = np.sqrt(rho) * z + np.sqrt(1 - rho) * eps
        active = alive & (t <= maturity)
        default_now = active & (asset < threshold)
        pays = active & ~default_now

        cf[:, t - 1] += (pays * principal * coupon).sum(axis=1)
        cf[:, t - 1] += (pays * (t == maturity) * principal).sum(axis=1)

        shock = PRICE_MEAN + PRICE_VOL * (PRICE_CORR * z + np.sqrt(1 - PRICE_CORR**2) * eta)
        collateral = principal / ltv * (1 + shock)
        recovery = np.clip(collateral * (1 - FIRE_SALE_COST), 0.0, principal)
        cf[:, t] += (default_now * recovery).sum(axis=1)

        loss += default_now * (principal - recovery)
        recovered += default_now * recovery
        defaulted |= default_now
        alive &= ~default_now

    invested = principal.sum()
    defaulted_principal = (defaulted * principal).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recovery_rate = np.where(defaulted_principal > 0,
                                 recovered.sum(axis=1) / defaulted_principal, np.nan)
    return {
        "irr": irr(cf, invested),
        "loss_pct": loss.sum(axis=1) / invested,
        "n_defaults": defaulted.sum(axis=1),
        "default_share": defaulted_principal / invested,
        "recovery_rate": recovery_rate,
    }


def summarise(res: dict) -> dict:
    loss = res["loss_pct"]
    var99 = np.quantile(loss, 0.99)
    return {
        "Expected loss": loss.mean(),
        "Mean IRR": res["irr"].mean(),
        "Median IRR": np.median(res["irr"]),
        "IRR 5th percentile": np.quantile(res["irr"], 0.05),
        "P(IRR < 0)": (res["irr"] < 0).mean(),
        "Credit VaR 99%": var99,
        "Expected shortfall 99%": loss[loss >= var99].mean(),
        "Mean defaults": res["n_defaults"].mean(),
        "P(at least 1 default)": (res["n_defaults"] > 0).mean(),
    }


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot(base: dict, runs: dict, contractual_irr: float, path: Path):
    colors = {0.0: "#2e8b57", RHO_BASE: "#1f4e79", 0.30: "#c0504d"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Hypothetical real estate private debt portfolio: Monte Carlo "
                 f"({N_SIM:,} scenarios, {N_LOANS} loans)", fontweight="bold")

    ax = axes[0]
    x_lo = np.quantile(base["irr"], 0.001) * 100 - 0.1
    x_hi = contractual_irr * 100 + 0.3
    ax.hist(base["irr"] * 100, bins=50, range=(x_lo, x_hi), color="#1f4e79", alpha=0.85)
    ax.axvline(contractual_irr * 100, color="black", ls="--", lw=1.2,
               label=f"No-default IRR {contractual_irr:.2%}")
    p5 = np.quantile(base["irr"], 0.05) * 100
    ax.axvline(p5, color="#c0504d", ls="--", lw=1.2, label=f"5th percentile {p5:.2f}%")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Portfolio IRR (%), axis cut at the 0.1th percentile")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"IRR distribution (correlation {RHO_BASE})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    top = max(np.quantile(r["loss_pct"], 0.999) for r in runs.values()) * 100
    bins = np.linspace(0, top, 50)
    for rho, r in runs.items():
        ax.hist(r["loss_pct"] * 100, bins=bins, histtype="step", lw=1.8, color=colors[rho],
                label=f"correlation {rho}  (mean loss {r['loss_pct'].mean():.2%})")
    ax.set_yscale("log")
    ax.set_xlabel("Credit loss (% of invested capital)")
    ax.set_ylabel("Number of scenarios (log scale)")
    ax.set_title("Loss distribution by default correlation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[2]
    mask = ~np.isnan(base["recovery_rate"])
    idx = np.flatnonzero(mask)[:3000]
    ax.scatter(base["default_share"][idx] * 100, base["recovery_rate"][idx] * 100,
               s=6, alpha=0.3, color="#1f4e79")
    ax.set_xlabel("Defaulted principal (% of portfolio)")
    ax.set_ylabel("Average recovery rate (%)")
    ax.set_title("Defaults vs recovery rate (scenarios with defaults)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    tape = make_tape(rng)
    OUT_DIR.mkdir(exist_ok=True)
    tape.to_csv(OUT_DIR / "loan_tape.csv", index=False)

    invested = tape.principal_chf_m.sum()
    w = tape.principal_chf_m / invested
    print("HYPOTHETICAL PORTFOLIO (randomly generated, not real data)")
    print(f"  Loans: {N_LOANS}   Invested: CHF {invested:.1f} m")
    print(f"  Weighted coupon: {(w * tape.coupon).sum():.2%}   "
          f"Weighted maturity: {(w * tape.maturity_years).sum():.2f} years   "
          f"Weighted LTV: {(w * tape.ltv).sum():.1%}")

    cf = contractual_cash_flows(tape)
    y = float(irr(cf, invested)[0])
    d = macaulay_duration(cf, y)
    print(f"  Contractual IRR (no defaults): {y:.2%}   Macaulay duration: {d:.2f} years   "
          f"Modified duration: {d / (1 + y):.2f}")

    # Check: the simulated default count must match the theory for any correlation
    # (each year every loan has default probability = PD, whatever the correlation).
    expected_defaults = (1 - (1 - tape.annual_pd) ** tape.maturity_years).sum()

    runs, rows = {}, {}
    for rho in RHO_SENSITIVITY:
        runs[rho] = simulate(tape, rho, np.random.default_rng(SEED + 1))
        rows[f"corr {rho}"] = summarise(runs[rho])
    summary = pd.DataFrame(rows)
    summary.round(5).to_csv(OUT_DIR / "mc_results.csv")

    print(f"\nCheck: expected defaults over the loan lives (theory) = {expected_defaults:.3f}, "
          f"simulated = " + ", ".join(f"{runs[r]['n_defaults'].mean():.3f} (corr {r})" for r in runs))

    show = summary.astype(object)
    pct_rows = ["Expected loss", "Mean IRR", "Median IRR", "IRR 5th percentile", "P(IRR < 0)",
                "Credit VaR 99%", "Expected shortfall 99%", "P(at least 1 default)"]
    for row in pct_rows:
        show.loc[row] = summary.loc[row].map("{:.2%}".format)
    show.loc["Mean defaults"] = summary.loc["Mean defaults"].map("{:.2f}".format)
    print("\n" + show.to_string())

    plot(runs[RHO_BASE], runs, y, OUT_DIR / "private_debt_mc.png")
    print(f"\nSaved loan_tape.csv, mc_results.csv and private_debt_mc.png in {OUT_DIR}/")


if __name__ == "__main__":
    main()
