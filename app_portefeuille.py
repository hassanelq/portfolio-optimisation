import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------
# Utils
# ----------------------------------------------------
FREQ = 252  # jours de bourse / an


def load_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = data.dropna()
    return data


def compute_return_stats_from_prices(prices, freq=FREQ):
    rets = prices.pct_change().dropna()
    mu = rets.mean() * freq
    cov = rets.cov() * freq
    return rets, mu, cov


def estimate_covariance(returns, method="empirical"):
    if method == "empirical":
        cov = returns.cov()
        return cov * FREQ
    elif method == "ledoit":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError:
            cov = returns.cov()
            return cov * FREQ
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
        return pd.DataFrame(cov * FREQ, index=returns.columns, columns=returns.columns)
    else:
        cov = returns.cov()
        return cov * FREQ


# ----------------------------------------------------
# Portfolio models
# ----------------------------------------------------
def equal_weighted_portfolio(assets):
    n = len(assets)
    return pd.Series(np.ones(n) / n, index=assets)


def optimize_global_min_variance(
    cov, min_weight=0.0, max_weight=1.0, allow_short=False
):
    n = len(cov)
    w = cp.Variable(n)
    cov_mat = cov.values

    risk = cp.quad_form(w, cov_mat)
    constraints = [cp.sum(w) == 1]

    if allow_short:
        constraints += [w >= -max_weight, w <= max_weight]
    else:
        constraints += [w >= min_weight, w <= max_weight]

    prob = cp.Problem(cp.Minimize(risk), constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    return pd.Series(np.array(w.value).flatten(), index=cov.index)


def optimize_mean_variance(
    mu,
    cov,
    risk_aversion=3.0,
    target_return=None,
    min_weight=0.0,
    max_weight=1.0,
    allow_short=False,
):
    n = len(mu)
    w = cp.Variable(n)
    mu_vec = mu.values
    cov_mat = cov.values

    risk = cp.quad_form(w, cov_mat)
    ret = mu_vec @ w

    constraints = [cp.sum(w) == 1]
    if allow_short:
        constraints += [w >= -max_weight, w <= max_weight]
    else:
        constraints += [w >= min_weight, w <= max_weight]

    # Handle target return or risk aversion
    if target_return is not None:
        constraints += [ret >= target_return]
        objective = cp.Minimize(risk)
    elif risk_aversion is not None:
        objective = cp.Maximize(ret - risk_aversion * risk)
    else:
        objective = cp.Minimize(risk)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    return pd.Series(np.array(w.value).flatten(), index=mu.index)


def optimize_tangency(mu, cov, min_weight=0.0, max_weight=1.0, allow_short=False):
    """
    max Sharpe ~ max mu^T x  s.c. x^T Σ x <= 1, contraintes de signe.
    Puis normalisation w = x / sum(x).
    """
    n = len(mu)
    x = cp.Variable(n)
    mu_vec = mu.values
    cov_mat = cov.values

    risk = cp.quad_form(x, cov_mat)
    ret = mu_vec @ x

    constraints = [risk <= 1]

    if allow_short:
        constraints += [x >= -max_weight, x <= max_weight]
    else:
        constraints += [x >= min_weight, x <= max_weight]

    prob = cp.Problem(cp.Maximize(ret), constraints)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if x.value is None:
        return None

    x_opt = np.array(x.value).flatten()
    if np.allclose(x_opt, 0):
        return None

    w = x_opt / np.sum(np.abs(x_opt))
    return pd.Series(w, index=mu.index)


def risk_parity_portfolio(cov, max_iter=1000, tol=1e-8):
    n = len(cov)
    cov_mat = cov.values

    vols = np.sqrt(np.diag(cov_mat))
    w = 1 / vols
    w = w / w.sum()

    for _ in range(max_iter):
        port_vol = np.sqrt(w @ cov_mat @ w)
        mrc = cov_mat @ w
        rc = w * mrc / port_vol
        target = port_vol / n
        w_new = w * target / rc
        w_new = w_new / w_new.sum()
        if np.allclose(w, w_new, rtol=tol):
            break
        w = w_new

    return pd.Series(w, index=cov.index)


# ----------------------------------------------------
# Black–Litterman
# ----------------------------------------------------
def compute_equilibrium_returns(returns, cov, market_weights, rf=0.0):
    """
    π = δ Σ w_mkt, with δ = (μ_mkt - rf) / σ_mkt^2
    """
    r_mkt_daily = returns @ market_weights
    mu_mkt = r_mkt_daily.mean() * FREQ
    var_mkt = (r_mkt_daily.var()) * FREQ
    if var_mkt <= 0:
        delta = 2.5
    else:
        delta = (mu_mkt - rf) / var_mkt
    pi = delta * cov.values @ market_weights
    return pd.Series(pi, index=cov.index), delta


def black_litterman(mu_prior_pi, cov, P, q, Omega, tau=0.05):
    """
    μ_BL = π + τ Σ Pᵀ (P τ Σ Pᵀ + Ω)⁻¹ (q − P π)
    """
    Sigma = cov.values
    pi = mu_prior_pi.values.reshape(-1, 1)
    P_mat = np.asarray(P)
    q_vec = np.asarray(q).reshape(-1, 1)
    Omega_mat = np.asarray(Omega)

    middle = np.linalg.inv(P_mat @ (tau * Sigma) @ P_mat.T + Omega_mat)
    adj = q_vec - P_mat @ pi
    mu_bl = pi + tau * Sigma @ P_mat.T @ middle @ adj
    mu_bl = mu_bl.flatten()
    return pd.Series(mu_bl, index=mu_prior_pi.index)


def build_bl_from_simple_views(returns, cov, rf, tau, view_dict, view_uncertainty_pct):
    """
    view_dict: {asset -> annual expected return (in decimal)}
    Omega: diag(σ_view^2) with σ_view in decimal.
    P: one row per view, identity on relevant asset.
    """
    assets = list(cov.index)
    market_w = equal_weighted_portfolio(assets).values
    pi, delta = compute_equilibrium_returns(returns, cov, market_w, rf=rf)

    active_assets = [a for a, v in view_dict.items() if v is not None]
    if len(active_assets) == 0:
        return pi

    K = len(active_assets)
    N = len(assets)
    P = np.zeros((K, N))
    q = np.zeros(K)
    for k, asset in enumerate(active_assets):
        idx = assets.index(asset)
        P[k, idx] = 1.0
        q[k] = view_dict[asset]

    sigma_view = view_uncertainty_pct / 100.0
    Omega = np.eye(K) * (sigma_view**2)

    mu_bl = black_litterman(pi, cov, P, q, Omega, tau=tau)
    return mu_bl


# ----------------------------------------------------
# Performance / risk metrics
# ----------------------------------------------------
def portfolio_performance(weights, mu, cov):
    w = weights.values
    mu_vec = mu.values
    cov_mat = cov.values
    ret = float(mu_vec @ w)
    vol = float(np.sqrt(w @ cov_mat @ w))
    sharpe = ret / vol if vol != 0 else np.nan
    return ret, vol, sharpe


def backtest_constant_weights(weights, returns):
    w = weights.values
    port_daily = (returns @ w).rename("Portfolio")
    cum = (1 + port_daily).cumprod()
    return port_daily, cum


def max_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    dd = (cum_returns / roll_max) - 1
    return dd.min()


def backtest_stats_from_series(port_daily, freq=FREQ):
    ann_ret = (1 + port_daily.mean()) ** freq - 1
    ann_vol = port_daily.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum = (1 + port_daily).cumprod()
    mdd = max_drawdown(cum)
    return ann_ret, ann_vol, sharpe, mdd


def herfindahl_index(weights):
    w = weights.values
    return np.sum(w**2)


def turnover_between_weights(w_prev, w_new):
    return np.sum(np.abs(w_new.values - w_prev.values))


# ----------------------------------------------------
# Rolling out-of-sample backtest
# ----------------------------------------------------
def rolling_backtest_models(
    returns,
    cov_method,
    model_configs,
    window_years=3,
    rebalance_months=3,
):
    assets = list(returns.columns)
    models = [
        name for name, config in model_configs.items() if config.get("enabled", True)
    ]

    backtest_rets = pd.DataFrame(index=returns.index, columns=models, dtype=float)
    last_weights = {m: None for m in models}
    turnover_acc = {m: 0.0 for m in models}
    n_rebals = {m: 0 for m in models}

    dates = returns.index
    start_date = dates[0]
    end_date = dates[-1]

    oos_start = start_date + pd.DateOffset(years=window_years)
    oos_start = dates[dates >= oos_start][0]

    rebal_dates = []
    cur = oos_start
    while cur <= end_date:
        cur_eff = dates[dates >= cur][0]
        rebal_dates.append(cur_eff)
        cur = cur + pd.DateOffset(months=rebalance_months)

    for i, rebal_date in enumerate(rebal_dates):
        train_start = rebal_date - pd.DateOffset(years=window_years)
        train_mask = (dates >= train_start) & (dates < rebal_date)
        train_rets = returns.loc[train_mask]
        if len(train_rets) < 60:
            continue

        mu_sample = train_rets.mean() * FREQ
        cov_est = estimate_covariance(train_rets, method=cov_method)

        # Build Black-Litterman mu if needed
        mu_bl = None
        bl_config = model_configs.get("Mean-Variance (BL µ)", {})
        if bl_config.get("enabled", False):
            mu_bl = build_bl_from_simple_views(
                train_rets,
                cov_est,
                bl_config["rf"],
                bl_config["tau"],
                bl_config["views"],
                bl_config["uncertainty_pct"],
            )

        if i < len(rebal_dates) - 1:
            hold_end = rebal_dates[i + 1] - pd.Timedelta(days=1)
        else:
            hold_end = end_date
        hold_mask = (dates >= rebal_date) & (dates <= hold_end)
        hold_rets = returns.loc[hold_mask]

        if hold_rets.empty:
            continue

        w_dict = {}

        # Equal-Weighted
        if model_configs.get("Equal-Weighted", {}).get("enabled", False):
            w_dict["Equal-Weighted"] = equal_weighted_portfolio(assets)

        # Min Variance
        mv_config = model_configs.get("Min Variance", {})
        if mv_config.get("enabled", False):
            w_dict["Min Variance"] = optimize_global_min_variance(
                cov_est,
                min_weight=mv_config["min_weight"],
                max_weight=mv_config["max_weight"],
                allow_short=mv_config["allow_short"],
            )

        # Mean-Variance (Sample µ)
        meanvar_config = model_configs.get("Mean-Variance (Sample µ)", {})
        if meanvar_config.get("enabled", False):
            w_dict["Mean-Variance (Sample µ)"] = optimize_mean_variance(
                mu_sample,
                cov_est,
                risk_aversion=meanvar_config["risk_aversion"],
                target_return=None,
                min_weight=meanvar_config["min_weight"],
                max_weight=meanvar_config["max_weight"],
                allow_short=meanvar_config["allow_short"],
            )

        # Tangency
        tan_config = model_configs.get("Tangency", {})
        if tan_config.get("enabled", False):
            # Use BL mu for tangency if available, otherwise sample mu
            mu_for_tan = mu_bl if mu_bl is not None else mu_sample
            w_dict["Tangency"] = optimize_tangency(
                mu_for_tan,
                cov_est,
                min_weight=tan_config["min_weight"],
                max_weight=tan_config["max_weight"],
                allow_short=tan_config["allow_short"],
            )

        # Risk Parity
        if model_configs.get("Risk Parity", {}).get("enabled", False):
            w_dict["Risk Parity"] = risk_parity_portfolio(cov_est)

        # Mean-Variance (BL µ)
        if bl_config.get("enabled", False) and mu_bl is not None:
            w_dict["Mean-Variance (BL µ)"] = optimize_mean_variance(
                mu_bl,
                cov_est,
                risk_aversion=bl_config["risk_aversion"],
                target_return=None,
                min_weight=bl_config["min_weight"],
                max_weight=bl_config["max_weight"],
                allow_short=bl_config["allow_short"],
            )

        for name, w in w_dict.items():
            if w is None:
                continue
            if last_weights[name] is not None:
                turnover_acc[name] += turnover_between_weights(last_weights[name], w)
            last_weights[name] = w
            n_rebals[name] += 1

            backtest_rets.loc[hold_mask, name] = hold_rets @ w.values

    return backtest_rets, turnover_acc, n_rebals


# ----------------------------------------------------
# Plots
# ----------------------------------------------------
def plot_efficient_frontier(
    frontier_vols, frontier_rets, tangency_point, rf, portfolios_dict, mu_used, cov_used
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frontier_vols * 100,
            y=frontier_rets * 100,
            mode="lines",
            name="Frontière Efficiente",
            line=dict(width=3),
        )
    )

    if tangency_point is not None and rf is not None:
        sigma_T, mu_T = tangency_point
        max_sigma = max(frontier_vols.max(), sigma_T) * 1.2
        xs = np.linspace(0.0, max_sigma, 50)
        slope = (mu_T - rf) / sigma_T if sigma_T > 0 else 0.0
        ys = rf + slope * xs
        fig.add_trace(
            go.Scatter(
                x=xs * 100,
                y=ys * 100,
                mode="lines",
                name="CML",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[sigma_T * 100],
                y=[mu_T * 100],
                mode="markers",
                name="Portefeuille tangent",
                marker=dict(size=12, color="red"),
            )
        )

    colors = ["green", "orange", "purple", "brown", "pink"]
    for idx, (name, w) in enumerate(portfolios_dict.items()):
        if w is None:
            continue
        r, v, s = portfolio_performance(w, mu_used, cov_used)
        fig.add_trace(
            go.Scatter(
                x=[v * 100],
                y=[r * 100],
                mode="markers",
                name=name,
                marker=dict(size=10, color=colors[idx % len(colors)]),
            )
        )

    fig.update_layout(
        title="Frontière efficiente, CML et portefeuilles",
        xaxis_title="Volatilité annualisée (%)",
        yaxis_title="Rendement annualisé (%)",
        template="plotly_white",
        height=550,
    )
    return fig


def plot_weights_bar(portfolios_dict):
    df = pd.DataFrame(
        {name: w.values for name, w in portfolios_dict.items() if w is not None},
        index=list(portfolios_dict.values())[0].index,
    )
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[col] * 100,
                name=col,
            )
        )
    fig.update_layout(
        title="Allocations (%)",
        xaxis_title="Actifs",
        yaxis_title="Poids (%)",
        barmode="group",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_cumulative_returns(df_cum):
    fig = go.Figure()
    for col in df_cum.columns:
        fig.add_trace(
            go.Scatter(
                x=df_cum.index,
                y=df_cum[col],
                mode="lines",
                name=col,
            )
        )
    fig.update_layout(
        title="Performance cumulée (out-of-sample)",
        xaxis_title="Date",
        yaxis_title="Valeur du portefeuille",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_correlation_matrix(returns):
    corr = returns.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Corrélation"),
        )
    )
    fig.update_layout(
        title="Matrice de corrélation", height=450, template="plotly_white"
    )
    return fig


def plot_risk_contribution(weights, cov):
    w = weights.values
    cov_mat = cov.values
    port_vol = np.sqrt(w @ cov_mat @ w)
    mrc = cov_mat @ w
    rc = w * mrc / port_vol
    rc_pct = rc / rc.sum() * 100

    fig = go.Figure(
        data=[
            go.Pie(
                labels=weights.index,
                values=rc_pct,
                hole=0.3,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Contribution au risque (%)", height=400, template="plotly_white"
    )
    return fig


# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimization (Markowitz + BL)", page_icon="📊", layout="wide"
)

st.title("📊 Portfolio Optimization – Markowitz, Risk Parity & Black–Litterman")

st.markdown(
    """
Application d'optimisation de portefeuille centrée sur :

- Mean–Variance (Markowitz) et Minimum Variance  
- Portefeuille tangent (Max Sharpe) et **Capital Market Line (CML)**  
- **Risk Parity**  
- **Black–Litterman** avec covariance robuste **Ledoit–Wolf**  
- Backtest **out-of-sample** (fenêtre glissante, rebalancement périodique)
"""
)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("📈 Données de marché", expanded=True):
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, V"
    tickers_str = st.text_input("Tickers (séparés par virgules)", default_tickers)
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", dt.date(2014, 1, 1))
    with col2:
        end_date = st.date_input("Fin", dt.date.today())

# Model selection
st.sidebar.markdown("### 🎯 Modèles de portefeuille")

use_equal_weighted = st.sidebar.checkbox("Equal-Weighted", value=True)

use_min_variance = st.sidebar.checkbox("Minimum Variance", value=True)
if use_min_variance:
    with st.sidebar.expander("⚙️ Paramètres Min Variance"):
        mv_min_weight = st.slider("Poids minimum (%)", 0, 20, 0, key="mv_min") / 100
        mv_max_weight = st.slider("Poids maximum (%)", 10, 100, 40, key="mv_max") / 100
        mv_allow_short = st.checkbox(
            "Autoriser les ventes à découvert", value=False, key="mv_short"
        )
else:
    mv_min_weight = 0.0
    mv_max_weight = 0.4
    mv_allow_short = False

use_mean_variance = st.sidebar.checkbox("Mean-Variance (Sample µ)", value=True)
if use_mean_variance:
    with st.sidebar.expander("⚙️ Paramètres Mean-Variance"):
        meanvar_min_weight = (
            st.slider("Poids minimum (%)", 0, 20, 0, key="meanvar_min") / 100
        )
        meanvar_max_weight = (
            st.slider("Poids maximum (%)", 10, 100, 40, key="meanvar_max") / 100
        )
        meanvar_allow_short = st.checkbox(
            "Autoriser les ventes à découvert", value=False, key="meanvar_short"
        )
        meanvar_risk_aversion = st.slider(
            "Aversion au risque λ", 0.1, 10.0, 3.0, 0.1, key="meanvar_lambda"
        )
else:
    meanvar_min_weight = 0.0
    meanvar_max_weight = 0.4
    meanvar_allow_short = False
    meanvar_risk_aversion = 3.0

use_tangency = st.sidebar.checkbox("Tangency (Max Sharpe)", value=True)
if use_tangency:
    with st.sidebar.expander("⚙️ Paramètres Tangency"):
        tan_min_weight = st.slider("Poids minimum (%)", 0, 20, 0, key="tan_min") / 100
        tan_max_weight = (
            st.slider("Poids maximum (%)", 10, 100, 40, key="tan_max") / 100
        )
        tan_allow_short = st.checkbox(
            "Autoriser les ventes à découvert", value=False, key="tan_short"
        )
        rf_cml = (
            st.number_input(
                "Taux sans risque annuel (pour CML) (%)",
                value=0.0,
                step=0.25,
                key="tan_rf",
            )
            / 100.0
        )
else:
    tan_min_weight = 0.0
    tan_max_weight = 0.4
    tan_allow_short = False
    rf_cml = 0.0

use_risk_parity = st.sidebar.checkbox("Risk Parity", value=True)

use_bl = st.sidebar.checkbox("Mean-Variance (Black-Litterman µ)", value=True)
if use_bl:
    with st.sidebar.expander("⚙️ Paramètres Black-Litterman"):
        cov_method = st.selectbox(
            "Estimateur de covariance",
            ["Empirical", "Ledoit–Wolf"],
            index=1,
            key="bl_cov",
        )
        cov_method_key = "empirical" if cov_method == "Empirical" else "ledoit"

        bl_min_weight = st.slider("Poids minimum (%)", 0, 20, 0, key="bl_min") / 100
        bl_max_weight = st.slider("Poids maximum (%)", 10, 100, 40, key="bl_max") / 100
        bl_allow_short = st.checkbox(
            "Autoriser les ventes à découvert", value=False, key="bl_short"
        )
        bl_risk_aversion = st.slider(
            "Aversion au risque λ", 0.1, 10.0, 3.0, 0.1, key="bl_lambda"
        )

        bl_rf = (
            st.number_input(
                "Taux sans risque annuel (%)", value=0.0, step=0.25, key="bl_rf"
            )
            / 100.0
        )
        bl_tau = st.slider("Paramètre τ", 0.01, 0.50, 0.05, 0.01, key="bl_tau")
        bl_uncertainty_pct = st.slider(
            "Incertitude sur les vues (écart-type, % annuel)",
            1.0,
            30.0,
            5.0,
            0.5,
            key="bl_uncertainty",
        )

        st.markdown("**Vues Black–Litterman (rendements annuels attendus, en %) :**")
        bl_views = {}
        for t in tickers:
            v = st.number_input(
                f"View sur {t} (laisser 0 pour aucune vue)",
                value=0.0,
                step=0.5,
                format="%.2f",
                key=f"bl_view_{t}",
            )
            bl_views[t] = v / 100.0 if v != 0.0 else None
else:
    cov_method_key = "ledoit"
    bl_min_weight = 0.0
    bl_max_weight = 0.4
    bl_allow_short = False
    bl_risk_aversion = 3.0
    bl_rf = 0.0
    bl_tau = 0.05
    bl_uncertainty_pct = 5.0
    bl_views = {}

with st.sidebar.expander("⏱ Backtest out-of-sample", expanded=True):
    window_years = st.slider(
        "Fenêtre de calibration (années)",
        1,
        10,
        3,
        help="Nombre d'années de données historiques utilisées pour calibrer les modèles à chaque rebalancement",
    )
    rebalance_months = st.slider(
        "Fréquence de rebalancement (mois)",
        1,
        12,
        3,
        help="Nombre de mois entre chaque rebalancement du portefeuille",
    )

run_button = st.sidebar.button("🚀 Lancer l'analyse", type="primary")

# ---------------- Main logic ----------------
# Initialize session state
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if run_button:
    st.session_state.analysis_complete = True

if not st.session_state.analysis_complete:
    st.info("Configure les paramètres puis clique sur **Lancer l'analyse**.")
    st.stop()

# Data loading
with st.spinner("Téléchargement des données..."):
    prices = load_price_data(tickers, start_date, end_date)

if prices.empty:
    st.error("Aucune donnée téléchargée. Vérifie tickers et dates.")
    st.stop()

returns = prices.pct_change().dropna()
mu_sample_full = returns.mean() * FREQ
cov_full = estimate_covariance(returns, method=cov_method_key)

mu_bl_full = None
if use_bl:
    mu_bl_full = build_bl_from_simple_views(
        returns,
        cov_full,
        bl_rf,
        bl_tau,
        bl_views,
        bl_uncertainty_pct,
    )

# Build model configurations
model_configs = {}

if use_equal_weighted:
    model_configs["Equal-Weighted"] = {"enabled": True}

if use_min_variance:
    model_configs["Min Variance"] = {
        "enabled": True,
        "min_weight": mv_min_weight,
        "max_weight": mv_max_weight,
        "allow_short": mv_allow_short,
    }

if use_mean_variance:
    model_configs["Mean-Variance (Sample µ)"] = {
        "enabled": True,
        "min_weight": meanvar_min_weight,
        "max_weight": meanvar_max_weight,
        "allow_short": meanvar_allow_short,
        "risk_aversion": meanvar_risk_aversion,
    }

if use_tangency:
    model_configs["Tangency"] = {
        "enabled": True,
        "min_weight": tan_min_weight,
        "max_weight": tan_max_weight,
        "allow_short": tan_allow_short,
    }

if use_risk_parity:
    model_configs["Risk Parity"] = {"enabled": True}

if use_bl:
    model_configs["Mean-Variance (BL µ)"] = {
        "enabled": True,
        "min_weight": bl_min_weight,
        "max_weight": bl_max_weight,
        "allow_short": bl_allow_short,
        "risk_aversion": bl_risk_aversion,
        "rf": bl_rf,
        "tau": bl_tau,
        "views": bl_views,
        "uncertainty_pct": bl_uncertainty_pct,
    }

st.success(f"Données chargées : {len(tickers)} actifs, {len(prices)} jours de données")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Données",
        "🎯 Optimisation in-sample",
        "📈 Backtest out-of-sample",
        "📉 Analyse du risque",
    ]
)

# ---------------- Tab 1: Data ----------------
with tab1:
    st.header("Analyse des données")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prix historiques")
        fig_prices = px.line(prices, title="Évolution des prix")
        fig_prices.update_layout(height=400)
        st.plotly_chart(fig_prices, use_container_width=True)

    with c2:
        st.subheader("Rendements annualisés (sample µ)")
        fig_mu = px.bar(mu_sample_full * 100, title="Rendements annualisés (%)")
        fig_mu.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_mu, use_container_width=True)

        if use_bl and mu_bl_full is not None:
            st.subheader("Rendements Black–Litterman (µ_BL)")
            fig_bl = px.bar(mu_bl_full * 100, title="Rendements BL (%)")
            fig_bl.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bl, use_container_width=True)

    st.subheader("Matrice de corrélation")
    fig_corr = plot_correlation_matrix(returns)
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- Tab 2: In-sample optimization ----------------
with tab2:
    st.header("Optimisation in-sample (Markowitz, Tangency, Risk Parity, BL)")

    # Efficient frontier with sample µ (for theory) - use mean-variance settings if enabled, otherwise use defaults
    if use_mean_variance:
        frontier_min_w = meanvar_min_weight
        frontier_max_w = meanvar_max_weight
        frontier_short = meanvar_allow_short
    else:
        # Use first available model's settings or defaults
        frontier_min_w = 0.0
        frontier_max_w = 0.4
        frontier_short = False

    target_rets = np.linspace(mu_sample_full.min(), mu_sample_full.max(), 50)
    frontier_vols = []
    frontier_rets = []
    for tr in target_rets:
        w = optimize_mean_variance(
            mu_sample_full,
            cov_full,
            risk_aversion=None,
            target_return=tr,
            min_weight=frontier_min_w,
            max_weight=frontier_max_w,
            allow_short=frontier_short,
        )
        if w is None:
            continue
        r, v, _ = portfolio_performance(w, mu_sample_full, cov_full)
        frontier_rets.append(r)
        frontier_vols.append(v)
    frontier_vols = np.array(frontier_vols)
    frontier_rets = np.array(frontier_rets)

    portfolios_insample = {}

    if use_equal_weighted:
        portfolios_insample["Equal-Weighted"] = equal_weighted_portfolio(
            mu_sample_full.index
        )

    if use_min_variance:
        portfolios_insample["Min Variance"] = optimize_global_min_variance(
            cov_full,
            min_weight=mv_min_weight,
            max_weight=mv_max_weight,
            allow_short=mv_allow_short,
        )

    if use_mean_variance:
        portfolios_insample["Mean-Variance (Sample µ)"] = optimize_mean_variance(
            mu_sample_full,
            cov_full,
            risk_aversion=meanvar_risk_aversion,
            target_return=None,
            min_weight=meanvar_min_weight,
            max_weight=meanvar_max_weight,
            allow_short=meanvar_allow_short,
        )

    if use_tangency:
        portfolios_insample["Tangency"] = optimize_tangency(
            mu_bl_full if (use_bl and mu_bl_full is not None) else mu_sample_full,
            cov_full,
            min_weight=tan_min_weight,
            max_weight=tan_max_weight,
            allow_short=tan_allow_short,
        )

    if use_risk_parity:
        portfolios_insample["Risk Parity"] = risk_parity_portfolio(cov_full)

    if use_bl and mu_bl_full is not None:
        portfolios_insample["Mean-Variance (BL µ)"] = optimize_mean_variance(
            mu_bl_full,
            cov_full,
            risk_aversion=bl_risk_aversion,
            target_return=None,
            min_weight=bl_min_weight,
            max_weight=bl_max_weight,
            allow_short=bl_allow_short,
        )

    # Tangency point for CML
    mu_used_for_tan = (
        mu_bl_full if (use_bl and mu_bl_full is not None) else mu_sample_full
    )
    w_tan = portfolios_insample.get("Tangency")
    if w_tan is not None:
        mu_T, sigma_T, _ = portfolio_performance(w_tan, mu_used_for_tan, cov_full)
        tangency_point = (sigma_T, mu_T)
    else:
        tangency_point = None

    fig_frontier = plot_efficient_frontier(
        frontier_vols,
        frontier_rets,
        tangency_point,
        rf_cml,
        portfolios_insample,
        mu_used_for_tan,
        cov_full,
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    st.subheader("Allocations in-sample")
    fig_w = plot_weights_bar(portfolios_insample)
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Statistiques in-sample")
    stats_rows = []
    for name, w in portfolios_insample.items():
        if w is None:
            continue
        r, v, s = portfolio_performance(w, mu_used_for_tan, cov_full)
        hhi = herfindahl_index(w)
        stats_rows.append(
            {
                "Portefeuille": name,
                "Rendement (%)": f"{r*100:.2f}",
                "Volatilité (%)": f"{v*100:.2f}",
                "Sharpe": f"{s:.2f}",
                "HHI (concentration)": f"{hhi:.3f}",
            }
        )
    df_stats_in = pd.DataFrame(stats_rows)
    st.dataframe(df_stats_in, hide_index=True, use_container_width=True)

# ---------------- Tab 3: OOS backtest ----------------
with tab3:
    st.header("Backtest out-of-sample (rolling window)")

    with st.spinner("Backtest rolling en cours..."):
        bt_rets, turnover_acc, n_rebals = rolling_backtest_models(
            returns,
            cov_method_key,
            model_configs,
            window_years=window_years,
            rebalance_months=rebalance_months,
        )

    bt_rets = bt_rets.dropna(how="all")
    cum_df = (1 + bt_rets).cumprod().dropna(how="all")

    st.subheader("Performance cumulée (OOS)")
    fig_cum = plot_cumulative_returns(cum_df)
    st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Statistiques OOS par modèle")
    rows = []
    for col in bt_rets.columns:
        s = bt_rets[col].dropna()
        if s.empty:
            continue
        ann_ret, ann_vol, sharpe, mdd = backtest_stats_from_series(s)
        t = turnover_acc.get(col, 0.0)
        k = n_rebals.get(col, 1)
        avg_turnover = t / k if k > 0 else np.nan
        rows.append(
            {
                "Portefeuille": col,
                "Rendement ann. (%)": f"{ann_ret*100:.2f}",
                "Volatilité ann. (%)": f"{ann_vol*100:.2f}",
                "Sharpe": f"{sharpe:.2f}",
                "Max Drawdown (%)": f"{mdd*100:.2f}",
                "Turnover moyen": f"{avg_turnover:.3f}",
            }
        )
    df_bt_stats = pd.DataFrame(rows)
    st.dataframe(df_bt_stats, hide_index=True, use_container_width=True)

# ---------------- Tab 4: Risk analysis ----------------
with tab4:
    st.header("Analyse du risque (in-sample)")

    # Get list of available portfolios (non-None)
    available_portfolios = [
        name for name, w in portfolios_insample.items() if w is not None
    ]

    if not available_portfolios:
        st.warning("Aucun portefeuille disponible pour l'analyse du risque.")
    else:
        selected_name = st.selectbox(
            "Choisir un portefeuille pour la décomposition du risque",
            available_portfolios,
            key="risk_portfolio_selector",
        )
        w_sel = portfolios_insample[selected_name]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Allocation du capital (%)")
            fig_alloc = go.Figure(
                data=[
                    go.Pie(
                        labels=w_sel.index,
                        values=w_sel.values * 100,
                        hole=0.3,
                        textinfo="label+percent",
                    )
                ]
            )
            fig_alloc.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_alloc, use_container_width=True)

        with c2:
            st.subheader("Contribution au risque")
            fig_rc = plot_risk_contribution(w_sel, cov_full)
            st.plotly_chart(fig_rc, use_container_width=True)

        w = w_sel.values
        cov_mat = cov_full.values
        port_vol = np.sqrt(w @ cov_mat @ w)
        mrc = cov_mat @ w
        rc = w * mrc / port_vol
        rc_pct = rc / rc.sum() * 100
        vol_ind = np.sqrt(np.diag(cov_mat)) * 100

        df_risk = pd.DataFrame(
            {
                "Actif": w_sel.index,
                "Poids (%)": w_sel.values * 100,
                "Vol. individuelle (%)": vol_ind,
                "Contribution au risque (%)": rc_pct,
            }
        )
        st.subheader("Détails de la décomposition du risque")
        st.dataframe(
            df_risk.style.format(
                {
                    "Poids (%)": "{:.2f}",
                    "Vol. individuelle (%)": "{:.2f}",
                    "Contribution au risque (%)": "{:.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

# ---------------- Recommendation ----------------
st.markdown("---")
st.header("🎯 Synthèse")

if (
    "Mean-Variance (BL µ)" in portfolios_insample
    and portfolios_insample["Mean-Variance (BL µ)"] is not None
):
    ref_name = "Mean-Variance (BL µ)"
else:
    ref_name = (
        "Tangency"
        if portfolios_insample.get("Tangency") is not None
        else "Mean-Variance (Sample µ)"
    )

w_ref = portfolios_insample[ref_name]
mu_used_final = mu_bl_full if (use_bl and mu_bl_full is not None) else mu_sample_full
r_ref, v_ref, s_ref = portfolio_performance(w_ref, mu_used_final, cov_full)
hhi_ref = herfindahl_index(w_ref)

st.markdown(
    f"""
**Portefeuille de référence (théorique in-sample)** : **{ref_name}**

- Rendement attendu : **{r_ref*100:.2f}%**  
- Volatilité attendue : **{v_ref*100:.2f}%**  
- Sharpe théorique : **{s_ref:.2f}**  
- Indice de concentration HHI : **{hhi_ref:.3f}**
"""
)

if ref_name in bt_rets.columns:
    s = bt_rets[ref_name].dropna()
    if not s.empty:
        ann_ret, ann_vol, sharpe, mdd = backtest_stats_from_series(s)
        st.markdown(
            f"""
**Backtest out-of-sample (rolling)** – {ref_name} :

- Rendement annualisé : **{ann_ret*100:.2f}%**  
- Volatilité annualisée : **{ann_vol*100:.2f}%**  
- Sharpe out-of-sample : **{sharpe:.2f}**  
- Max Drawdown : **{mdd*100:.2f}%**
"""
        )
