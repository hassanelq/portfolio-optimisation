import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ----------------------------------------------------
# Utilitaires financiers
# ----------------------------------------------------
def load_price_data(tickers, start, end):
    """Télécharge les données de prix"""
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = data.dropna()
    return data


def compute_return_stats(prices, freq=252):
    """Calcule les rendements et statistiques"""
    rets = prices.pct_change().dropna()
    mu = rets.mean() * freq
    cov = rets.cov() * freq
    return rets, mu, cov


def shrink_covariance(cov, alpha=0.2):
    """Shrinkage vers la matrice diagonale"""
    diag = np.diag(np.diag(cov.values))
    cov_shrink = (1 - alpha) * cov.values + alpha * diag
    return pd.DataFrame(cov_shrink, index=cov.index, columns=cov.columns)


# ----------------------------------------------------
# Méthodes d'optimisation de portefeuille
# ----------------------------------------------------
def optimize_mean_variance(
    mu,
    cov,
    target_return=None,
    risk_aversion=None,
    min_weight=0.0,
    max_weight=1.0,
    allow_short=False,
):
    """Optimisation Moyenne-Variance classique"""
    n = len(mu)
    w = cp.Variable(n)

    risk = cp.quad_form(w, cov.values)
    ret = mu.values @ w

    constraints = [cp.sum(w) == 1]

    if allow_short:
        constraints += [w >= -max_weight, w <= max_weight]
    else:
        constraints += [w >= min_weight, w <= max_weight]

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
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    return pd.Series(np.array(w.value).flatten(), index=mu.index)


def optimize_max_sharpe(mu, cov, min_weight=0.0, max_weight=1.0, allow_short=False):
    """Portefeuille tangent (Max Sharpe)"""
    n = len(mu)
    w = cp.Variable(n)

    risk = cp.quad_form(w, cov.values)
    ret = mu.values @ w

    # Contraintes standard de portefeuille
    constraints = [cp.sum(w) == 1]

    if allow_short:
        constraints += [w >= -max_weight, w <= max_weight]
    else:
        constraints += [w >= min_weight, w <= max_weight]

    # Objectif: maximiser le Sharpe ratio
    # On maximise ret / sqrt(risk), équivalent à maximiser ret sous contrainte risk <= 1
    # Mais pour un vrai max Sharpe, on doit résoudre le problème différemment

    # Approche alternative: minimiser risk - kappa * ret où kappa est grand
    # Ou utiliser la formulation: max ret sous contrainte sqrt(risk) <= sigma_target

    # Meilleure approche: résoudre directement max (mu'w) / sqrt(w'Σw)
    # Ceci n'est pas convexe, mais on peut le transformer

    # Transformation: on maximise mu'w sujet à w'Σw <= sigma_target^2 et sum(w)=1
    # Puis on cherche le sigma_target qui donne le meilleur Sharpe

    # Approche standard: utiliser la formulation avec variable auxiliaire kappa
    kappa = cp.Variable()
    y = cp.Variable(n)

    # y = kappa * w, donc w = y / kappa
    # Sharpe = mu'w / sqrt(w'Σw) = (mu'y/kappa) / sqrt(y'Σy/kappa^2) = mu'y / sqrt(y'Σy)
    # On maximise mu'y sujet à y'Σy <= 1, kappa >= 0, sum(y) = kappa

    constraints_sharpe = [
        cp.quad_form(y, cov.values) <= 1,
        cp.sum(y) == kappa,
        kappa >= 0,
    ]

    if allow_short:
        constraints_sharpe += [y >= -max_weight * kappa, y <= max_weight * kappa]
    else:
        constraints_sharpe += [y >= min_weight * kappa, y <= max_weight * kappa]

    objective = cp.Maximize(mu.values @ y)
    prob = cp.Problem(objective, constraints_sharpe)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            prob.solve(solver=cp.CLARABEL, verbose=False)

    if y.value is None or kappa.value is None or kappa.value <= 1e-10:
        return None

    # Récupérer les poids: w = y / kappa
    w_opt = np.array(y.value).flatten() / kappa.value

    # Normaliser pour être sûr que sum(w) = 1
    w_opt = w_opt / np.sum(w_opt)

    return pd.Series(w_opt, index=mu.index)


def optimize_global_min_variance(
    cov, min_weight=0.0, max_weight=1.0, allow_short=False
):
    """Global Minimum Variance Portfolio (GMVP)"""
    n = len(cov)
    w = cp.Variable(n)

    risk = cp.quad_form(w, cov.values)

    constraints = [cp.sum(w) == 1]

    if allow_short:
        constraints += [w >= -max_weight, w <= max_weight]
    else:
        constraints += [w >= min_weight, w <= max_weight]

    prob = cp.Problem(cp.Minimize(risk), constraints)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    return pd.Series(np.array(w.value).flatten(), index=cov.index)


def equal_weighted_portfolio(assets):
    """Equal-Weighted Portfolio (EWP)"""
    n = len(assets)
    return pd.Series(np.ones(n) / n, index=assets)


def inverse_volatility_portfolio(cov):
    """Inverse-Volatility Portfolio (IVP)"""
    vols = np.sqrt(np.diag(cov.values))
    inv_vols = 1 / vols
    weights = inv_vols / inv_vols.sum()
    return pd.Series(weights, index=cov.index)


def most_diversified_portfolio(mu, cov, min_weight=0.0, max_weight=1.0):
    """Most Diversified Portfolio (MDP)"""
    n = len(cov)
    w = cp.Variable(n)

    vols = np.sqrt(np.diag(cov.values))

    # Diversification ratio = (w^T σ) / sqrt(w^T Σ w)
    # On maximise w^T σ sous contrainte w^T Σ w <= 1
    numerator = vols @ w
    denominator = cp.quad_form(w, cov.values)

    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight, denominator <= 1]

    prob = cp.Problem(cp.Maximize(numerator), constraints)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    weights = np.array(w.value).flatten()
    weights = weights / weights.sum()  # Normalisation
    return pd.Series(weights, index=cov.index)


def risk_parity_portfolio(cov, max_iter=1000, tol=1e-8):
    """Risk Parity Portfolio - contribution égale au risque"""
    n = len(cov)

    # Initialisation avec inverse volatility
    vols = np.sqrt(np.diag(cov.values))
    weights = 1 / vols
    weights = weights / weights.sum()

    cov_matrix = cov.values

    for _ in range(max_iter):
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / port_vol

        target_risk = port_vol / n

        # Mise à jour des poids
        weights_new = weights * target_risk / risk_contrib
        weights_new = weights_new / weights_new.sum()

        if np.allclose(weights, weights_new, rtol=tol):
            break

        weights = weights_new

    return pd.Series(weights, index=cov.index)


# ----------------------------------------------------
# Frontière efficiente
# ----------------------------------------------------
def monte_carlo_portfolios(
    mu, cov, num_simulations=10000, min_weight=0.0, max_weight=1.0, allow_short=False
):
    """Génère des portefeuilles aléatoires par Monte Carlo"""
    n = len(mu)
    results = []

    while len(results) < num_simulations:
        # Générer des poids aléatoires avec méthode Dirichlet pour garantir sum=1
        if allow_short:
            weights = np.random.uniform(-max_weight, max_weight, n)
            weights = weights / weights.sum()
        else:
            # Méthode Dirichlet pour générer des poids qui somment à 1
            weights = np.random.dirichlet(np.ones(n))
            # Ajuster pour respecter min/max weight
            if min_weight > 0 or max_weight < 1:
                weights = weights * (max_weight - min_weight) + min_weight
                weights = weights / weights.sum()  # Re-normaliser

        # Vérifier les contraintes min/max
        if np.any(weights < min_weight - 1e-6) or np.any(weights > max_weight + 1e-6):
            continue

        # Calculer performance
        port_ret = mu.values @ weights
        port_vol = np.sqrt(weights @ cov.values @ weights)
        port_sharpe = port_ret / port_vol if port_vol > 0 else 0

        results.append(
            {"return": port_ret, "volatility": port_vol, "sharpe": port_sharpe}
        )

    return pd.DataFrame(results)


def compute_efficient_frontier(
    mu, cov, num_points=100, min_weight=0.0, max_weight=1.0, allow_short=False
):
    """Calcule la frontière efficiente"""
    min_ret = mu.min()
    max_ret = mu.max()

    target_returns = np.linspace(min_ret, max_ret, num_points)
    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_returns:
        weights = optimize_mean_variance(
            mu,
            cov,
            target_return=target,
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
        )

        if weights is not None:
            port_ret, port_vol, _ = portfolio_performance(weights, mu, cov)
            frontier_rets.append(port_ret)
            frontier_vols.append(port_vol)
            frontier_weights.append(weights)

    return np.array(frontier_vols), np.array(frontier_rets), frontier_weights


# ----------------------------------------------------
# Fonctions de performance
# ----------------------------------------------------
def portfolio_performance(weights, mu, cov):
    """Calcule la performance d'un portefeuille"""
    w = weights.values
    mu_vec = mu.values
    cov_mat = cov.values

    port_ret = float(mu_vec @ w)
    port_vol = float(np.sqrt(w @ cov_mat @ w))
    sharpe = port_ret / port_vol if port_vol != 0 else np.nan

    return port_ret, port_vol, sharpe


def backtest_constant_weights(weights, returns):
    """Backtest avec poids constants"""
    w = weights.values
    port_daily = (returns @ w).rename("Portfolio")
    cum_returns = (1 + port_daily).cumprod()
    return port_daily, cum_returns


def max_drawdown(cum_returns):
    """Calcule le drawdown maximum"""
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns / roll_max) - 1
    return drawdown.min()


def backtest_stats(port_daily, freq=252):
    """Calcule les statistiques de backtest"""
    ann_ret = (1 + port_daily.mean()) ** freq - 1
    ann_vol = port_daily.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum = (1 + port_daily).cumprod()
    mdd = max_drawdown(cum)
    return ann_ret, ann_vol, sharpe, mdd


# ----------------------------------------------------
# Visualisations
# ----------------------------------------------------
def plot_efficient_frontier(
    frontier_vols, frontier_rets, portfolios_dict, mu, cov, mc_data=None
):
    """Trace la frontière efficiente avec les portefeuilles et Monte Carlo"""
    fig = go.Figure()

    # Monte Carlo simulations (en PREMIER pour être en arrière-plan visuellement)
    if mc_data is not None and not mc_data.empty:
        fig.add_trace(
            go.Scatter(
                x=mc_data["volatility"] * 100,
                y=mc_data["return"] * 100,
                mode="markers",
                name="Portefeuilles Aléatoires (Monte Carlo)",
                marker=dict(
                    size=3,
                    color=mc_data["sharpe"],
                    colorscale="Viridis",
                    showscale=False,  # Retirer la légende du gradient
                    opacity=0.3,
                    line=dict(width=0),
                ),
                hovertemplate="<b>Monte Carlo</b><br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
            )
        )

    # Frontière efficiente (au-dessus de Monte Carlo)
    fig.add_trace(
        go.Scatter(
            x=frontier_vols * 100,
            y=frontier_rets * 100,
            mode="lines",
            name="Frontière Efficiente (Optimisation)",
            line=dict(color="blue", width=4),
            hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
        )
    )

    # Portefeuilles individuels (au-dessus de tout)
    colors = ["red", "green", "orange", "purple", "brown", "pink", "cyan"]
    markers = ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"]

    for idx, (name, weights) in enumerate(portfolios_dict.items()):
        if weights is not None:
            ret, vol, sharpe = portfolio_performance(weights, mu, cov)
            fig.add_trace(
                go.Scatter(
                    x=[vol * 100],
                    y=[ret * 100],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=15,
                        color=colors[idx % len(colors)],
                        symbol=markers[idx % len(markers)],
                        line=dict(width=2, color="white"),
                    ),
                    hovertemplate=f"<b>{name}</b><br>Vol: %{{x:.2f}}%<br>Ret: %{{y:.2f}}%<br>Sharpe: {sharpe:.2f}<extra></extra>",
                    hoverlabel=dict(namelength=-1),
                )
            )

    # Actifs individuels
    for i, asset in enumerate(mu.index):
        asset_ret = mu[asset] * 100
        asset_vol = np.sqrt(cov.loc[asset, asset]) * 100
        fig.add_trace(
            go.Scatter(
                x=[asset_vol],
                y=[asset_ret],
                mode="markers+text",
                name=asset,
                text=[asset],
                textposition="top center",
                marker=dict(size=10, color="gray", symbol="circle"),
                hovertemplate=f"<b>{asset}</b><br>Vol: %{{x:.2f}}%<br>Ret: %{{y:.2f}}%<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Frontière Efficiente et Portefeuilles",
        xaxis_title="Volatilité Annualisée (%)",
        yaxis_title="Rendement Annualisé (%)",
        hovermode="closest",
        height=600,
        template="plotly_white",
    )

    return fig


def plot_weights_comparison(portfolios_dict):
    """Compare les poids des différents portefeuilles"""
    weights_df = pd.DataFrame(
        {
            name: weights.values
            for name, weights in portfolios_dict.items()
            if weights is not None
        },
        index=list(portfolios_dict.values())[0].index,
    )

    fig = go.Figure()

    for portfolio in weights_df.columns:
        fig.add_trace(
            go.Bar(
                name=portfolio,
                x=weights_df.index,
                y=weights_df[portfolio] * 100,
                text=weights_df[portfolio].apply(lambda x: f"{x*100:.1f}%"),
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Comparaison des Allocations",
        xaxis_title="Actifs",
        yaxis_title="Poids (%)",
        barmode="group",
        height=500,
        template="plotly_white",
    )

    return fig


def plot_cumulative_returns(returns_dict):
    """Trace les rendements cumulés"""
    fig = go.Figure()

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"]

    for idx, (name, cum_ret) in enumerate(returns_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name=name,
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Performance Cumulée des Portefeuilles",
        xaxis_title="Date",
        yaxis_title="Valeur du Portefeuille",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )

    return fig


def plot_correlation_matrix(cov, returns):
    """Matrice de corrélation interactive"""
    corr = returns.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Corrélation"),
        )
    )

    fig.update_layout(
        title="Matrice de Corrélation", height=500, template="plotly_white"
    )

    return fig


def plot_risk_contribution(weights, cov):
    """Contribution au risque de chaque actif"""
    w = weights.values
    cov_mat = cov.values

    port_vol = np.sqrt(w @ cov_mat @ w)
    marginal_contrib = cov_mat @ w
    risk_contrib = w * marginal_contrib / port_vol
    risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

    fig = go.Figure(
        data=[
            go.Pie(
                labels=weights.index,
                values=risk_contrib_pct,
                hole=0.3,
                textinfo="label+percent",
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Contribution au Risque Total", height=400, template="plotly_white"
    )

    return fig


# ----------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimization Pro",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Portfolio Optimization Professional Tool")
st.markdown(
    """
Application avancée d'optimisation de portefeuille avec multiples stratégies et visualisations interactives.
"""
)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Section 1: Données
with st.sidebar.expander("📈 Données de Marché", expanded=True):
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, V"
    tickers_str = st.text_input("Tickers (séparés par virgules)", default_tickers)
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", dt.date(2018, 1, 1))
    with col2:
        end_date = st.date_input("Fin", dt.date.today())

# Section 2: Méthodes
with st.sidebar.expander("🎯 Méthodes d'Optimisation", expanded=True):
    st.markdown("**Sélectionne les méthodes:**")

    # Method selection with checkboxes
    method_equal = st.checkbox("Equal-Weighted", value=True)
    method_inverse_vol = st.checkbox("Inverse-Volatility", value=True)
    method_gmvp = st.checkbox("Global Minimum Variance", value=True)
    method_mean_var = st.checkbox("Mean-Variance (Target Return)", value=True)
    method_max_sharpe = st.checkbox("Max Sharpe (Tangency)", value=True)
    method_mdp = st.checkbox("Most Diversified", value=True)
    method_risk_parity = st.checkbox("Risk Parity", value=True)

    # Parameters for Mean-Variance
    target_return = None
    risk_aversion = None

    if method_mean_var:
        st.markdown("**Paramètres Mean-Variance:**")
        target_return = (
            st.slider(
                "📊 Rendement cible annuel (%)",
                0,
                50,
                30,
                help="Minimise le risque pour atteindre ce rendement cible",
            )
            / 100
        )

        use_risk_aversion = st.checkbox(
            "Utiliser l'aversion au risque",
            value=True,
            help="Si activé, utilise l'approche par aversion au risque au lieu du rendement cible",
        )

        if use_risk_aversion:
            risk_aversion = st.slider(
                "⚠️ Coefficient d'aversion au risque (λ)",
                0.1,
                10.0,
                2.0,
                0.1,
                help="Plus λ est élevé, plus le portefeuille sera conservateur. Formule: max(μ'w - λ·w'Σw)",
            )
            target_return = (
                None  # Si on utilise risk_aversion, on n'utilise pas target_return
            )

# Section 3: Contraintes
with st.sidebar.expander("🔒 Contraintes", expanded=True):
    allow_short = st.checkbox("Autoriser short selling", value=True)
    min_weight = st.slider("Poids minimum (%)", 0, 20, 0) / 100
    max_weight = st.slider("Poids maximum (%)", 10, 100, 30) / 100

    use_shrinkage = st.checkbox("Utiliser covariance robuste (shrinkage)", value=True)
    if use_shrinkage:
        shrink_alpha = st.slider("Paramètre shrinkage", 0.0, 1.0, 0.2, 0.05)

# Section 4: Visualisation
with st.sidebar.expander("📊 Options de Visualisation", expanded=True):
    show_frontier = st.checkbox("Afficher frontière efficiente", value=True)
    show_correlation = st.checkbox("Afficher matrice de corrélation", value=True)
    show_risk_contrib = st.checkbox("Afficher contribution au risque", value=True)

run_button = st.sidebar.button("🚀 Lancer l'Optimisation", type="primary")

# Corps principal
if not run_button:
    st.info(
        "👈 Configure les paramètres dans la barre latérale et clique sur **Lancer l'Optimisation**"
    )

    # Afficher des informations sur les méthodes
    st.markdown("### 📚 Méthodes Disponibles")
    st.markdown("---")

    st.markdown(
        """
        #### 1️⃣ **Equal-Weighted Portfolio (EWP)**
        
        **Principe:** Allocation naïve avec poids égaux pour tous les actifs.
        
        **Formule:**
        $$w_i = \\frac{1}{N} \\quad \\forall i$$
        
        **Problème d'optimisation:** Aucun (allocation fixe)
        
        **Intuition:** 
        - Stratégie la plus simple: diversification maximale sans optimisation
        - Ignore complètement les corrélations et les rendements attendus
        - Surprenamment efficace en pratique (paradoxe de diversification)
        - Utilisé comme benchmark pour évaluer les stratégies sophistiquées
        
        **Avantages:** Simplicité, pas d'estimation de paramètres, faible turnover
        
        **Inconvénients:** Ignore le risque et les rendements individuels
        
        ---
        
        #### 2️⃣ **Inverse-Volatility Portfolio (IVP)**
        
        **Principe:** Poids inversement proportionnels à la volatilité individuelle.
        
        **Formule:**
        $$w_i = \\frac{1/\\sigma_i}{\\sum_{j=1}^{N} 1/\\sigma_j}$$
        
        où $\\sigma_i = \\sqrt{\\Sigma_{ii}}$ est la volatilité de l'actif $i$
        
        **Problème d'optimisation:** Aucun (allocation heuristique)
        
        **Intuition:**
        - Alloue plus de capital aux actifs moins volatiles
        - Première prise en compte du risque (mais uniquement le risque individuel)
        - Ignore les corrélations entre actifs
        - Facile à calculer et à interpréter
        
        **Avantages:** Simple, réduction du risque par rapport à EWP
        
        **Inconvénients:** Ignore les corrélations et les rendements attendus
        
        ---
        
        #### 3️⃣ **Global Minimum Variance Portfolio (GMVP)**
        
        **Principe:** Minimise le risque total du portefeuille sans considération de rendement.
        
        **Problème d'optimisation:**
        $$\\min_w \\quad w^T \\Sigma w$$
        $$\\text{s.t.} \\quad \\sum_{i=1}^{N} w_i = 1, \\quad w_i \\geq 0$$
        
        **Formule analytique (sans contraintes):**
        $$w = \\frac{\\Sigma^{-1} \\mathbf{1}}{\\mathbf{1}^T \\Sigma^{-1} \\mathbf{1}}$$
        
        **Intuition:**
        - Premier portefeuille sur la frontière efficiente
        - Exploite les corrélations pour minimiser la variance totale
        - Portefeuille le plus stable et conservateur
        - Point de départ de la théorie moderne du portefeuille (Markowitz, 1952)
        
        **Avantages:** Risque minimal, insensible aux erreurs d'estimation de rendements
        
        **Inconvénients:** Rendement attendu potentiellement faible, sensible à l'estimation de la covariance
        
        ---
        
        #### 4️⃣ **Mean-Variance (Target Return)**
        
        **Principe:** Minimise le risque pour un niveau de rendement cible (approche de Markowitz).
        
        **Problème d'optimisation:**
        $$\\min_w \\quad w^T \\Sigma w$$
        $$\\text{s.t.} \\quad w^T \\mu \\geq r_{target}, \\quad \\sum_{i=1}^{N} w_i = 1, \\quad w_i \\geq 0$$
        
        **Variante (maximisation d'utilité):**
        $$\\max_w \\quad w^T \\mu - \\lambda w^T \\Sigma w$$
        
        où $\\lambda$ est le coefficient d'aversion au risque
        
        **Intuition:**
        - Approche classique de Markowitz (Prix Nobel 1990)
        - Trade-off explicite entre rendement et risque
        - Génère des portefeuilles sur la frontière efficiente
        - Nécessite l'estimation de $\\mu$ et $\\Sigma$
        
        **Avantages:** Équilibre personnalisable risque/rendement, base théorique solide
        
        **Inconvénients:** Très sensible aux erreurs d'estimation de $\\mu$, peut donner des poids extrêmes
        
        ---
        
        #### 5️⃣ **Max Sharpe (Tangency Portfolio)**
        
        **Principe:** Maximise le ratio de Sharpe (rendement par unité de risque).
        
        **Problème d'optimisation:**
        $$\\max_w \\quad \\frac{w^T \\mu}{\\sqrt{w^T \\Sigma w}}$$
        $$\\text{s.t.} \\quad \\sum_{i=1}^{N} w_i = 1, \\quad w_i \\geq 0$$
        
        **Transformation convexe (problème équivalent):**
        $$\\max_x \\quad \\mu^T x \\quad \\text{s.t.} \\quad x^T \\Sigma x \\leq 1, \\quad x \\geq 0$$
        puis normaliser: $w = x / \\sum x_i$
        
        **Formule analytique (sans contraintes):**
        $$w = \\frac{\\Sigma^{-1} \\mu}{\\mathbf{1}^T \\Sigma^{-1} \\mu}$$
        
        **Intuition:**
        - Portefeuille tangent: intersection de la frontière efficiente et de la ligne de marché des capitaux
        - Optimal pour un investisseur qui peut emprunter/prêter au taux sans risque
        - Maximise le rendement excédentaire par unité de risque
        - Portefeuille le plus "efficient" selon le CAPM
        
        **Avantages:** Meilleur ratio risque/rendement, justification théorique forte
        
        **Inconvénients:** Très sensible aux erreurs d'estimation, concentré sur peu d'actifs
        
        ---
        
        #### 6️⃣ **Most Diversified Portfolio (MDP)**
        
        **Principe:** Maximise le ratio de diversification.
        
        **Ratio de diversification:**
        $$DR(w) = \\frac{w^T \\sigma}{\\sqrt{w^T \\Sigma w}}$$
        
        où $\\sigma = [\\sigma_1, \\dots, \\sigma_N]^T$ est le vecteur des volatilités
        
        **Problème d'optimisation:**
        $$\\max_w \\quad \\frac{\\sum_{i=1}^{N} w_i \\sigma_i}{\\sqrt{w^T \\Sigma w}}$$
        $$\\text{s.t.} \\quad \\sum_{i=1}^{N} w_i = 1, \\quad w_i \\geq 0$$
        
        **Intuition:**
        - Cherche à maximiser les bénéfices de la diversification
        - Favorise les actifs peu corrélés entre eux
        - Équilibre entre volatilité individuelle et corrélations
        - Robuste aux erreurs d'estimation de rendements (n'utilise que $\\Sigma$)
        - Surperforme souvent le Max Sharpe en out-of-sample
        
        **Avantages:** Excellente diversification, robuste, peu concentré
        
        **Inconvénients:** Ignore les rendements attendus, calcul plus complexe
        
        ---
        
        #### 7️⃣ **Risk Parity Portfolio**
        
        **Principe:** Égalise les contributions au risque de chaque actif.
        
        **Contribution au risque de l'actif $i$:**
        $$RC_i = w_i \\frac{\\partial \\sigma_p}{\\partial w_i} = w_i \\frac{(\\Sigma w)_i}{\\sigma_p}$$
        
        **Condition d'équilibre:**
        $$RC_1 = RC_2 = \\dots = RC_N = \\frac{\\sigma_p}{N}$$
        
        **Problème d'optimisation (forme implicite):**
        Résolution numérique itérative pour satisfaire:
        $$w_i \\cdot (\\Sigma w)_i = \\text{constant} \\quad \\forall i$$
        
        **Intuition:**
        - Chaque actif contribue équitablement au risque total
        - Déconcentre le risque (contrairement à Max Sharpe qui peut concentrer)
        - Populaire chez les hedge funds et gestionnaires institutionnels
        - Extension de l'approche "60/40" actions/obligations
        - Allocation plus stable dans le temps
        
        **Avantages:** Diversification du risque, stabilité, performance out-of-sample
        
        **Inconvénients:** Pas de solution analytique fermée, ignore les rendements attendus
        
        ---
        
        ### 📊 **Résumé Comparatif**
        
        | Méthode | Paramètres requis |
        |---------|-------------------|
        | Equal-Weighted | Aucun |
        | Inverse-Volatility | $\\Sigma$ (diagonal) |
        | GMVP | $\\Sigma$ |
        | Mean-Variance | $\\mu$, $\\Sigma$ 
        | Max Sharpe | $\\mu$, $\\Sigma$ |
        | MDP | $\\Sigma$ |
        | Risk Parity |  $\\Sigma$ |
        
        **Note:** Les méthodes qui n'utilisent pas $\\mu$ (rendements attendus) sont généralement plus robustes car 
        $\\mu$ est très difficile à estimer avec précision.
        """
    )

    st.stop()

# Chargement des données
with st.spinner("📥 Téléchargement des données..."):
    try:
        prices = load_price_data(tickers, start_date, end_date)

        if prices.empty:
            st.error("❌ Aucune donnée téléchargée. Vérifie les tickers et les dates.")
            st.stop()

        returns, mu, cov = compute_return_stats(prices)

        if use_shrinkage:
            cov_used = shrink_covariance(cov, alpha=shrink_alpha)
        else:
            cov_used = cov

        st.success(f"✅ Données chargées: {len(tickers)} actifs, {len(prices)} jours")
    except Exception as e:
        st.error(f"❌ Erreur lors du téléchargement: {str(e)}")
        st.stop()

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Analyse", "🎯 Portefeuilles", "📈 Performance", "📉 Risque"]
)

with tab1:
    st.header("Analyse des Données")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prix Historiques")
        fig_prices = px.line(prices, title="Évolution des Prix")
        fig_prices.update_layout(height=400)
        st.plotly_chart(fig_prices, use_container_width=True)

    with col2:
        st.subheader("Rendements Espérés Annuels")
        fig_returns = px.bar(mu * 100, title="Rendements Annualisés (%)")
        fig_returns.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_returns, use_container_width=True)

    if show_correlation:
        st.subheader("Matrice de Corrélation")
        fig_corr = plot_correlation_matrix(cov_used, returns)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.header("Optimisation des Portefeuilles")

    # Optimisation
    portfolios = {}

    with st.spinner("🔄 Optimisation en cours..."):
        if method_max_sharpe:
            portfolios["Max Sharpe"] = optimize_max_sharpe(
                mu, cov_used, min_weight, max_weight, allow_short
            )

        if method_gmvp:
            portfolios["Min Variance"] = optimize_global_min_variance(
                cov_used, min_weight, max_weight, allow_short
            )

        if method_equal:
            portfolios["Equal-Weighted"] = equal_weighted_portfolio(mu.index)

        if method_inverse_vol:
            portfolios["Inverse-Volatility"] = inverse_volatility_portfolio(cov_used)

        if method_mdp:
            portfolios["Most Diversified"] = most_diversified_portfolio(
                mu, cov_used, min_weight, max_weight
            )

        if method_risk_parity:
            portfolios["Risk Parity"] = risk_parity_portfolio(cov_used)

        if method_mean_var:
            if risk_aversion is not None:
                # Utiliser l'approche par aversion au risque
                portfolios["Mean-Variance (Risk Aversion)"] = optimize_mean_variance(
                    mu,
                    cov_used,
                    risk_aversion=risk_aversion,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    allow_short=allow_short,
                )
            elif target_return is not None:
                # Utiliser l'approche par rendement cible
                portfolios["Mean-Variance (Target Return)"] = optimize_mean_variance(
                    mu,
                    cov_used,
                    target_return=target_return,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    allow_short=allow_short,
                )

    # Filtrer les portefeuilles None
    portfolios = {k: v for k, v in portfolios.items() if v is not None}

    if not portfolios:
        st.error(
            "❌ Aucun portefeuille n'a pu être optimisé. Essaie de modifier les contraintes."
        )
        st.stop()

    # Afficher la frontière efficiente
    if show_frontier:
        st.subheader("Frontière Efficiente")
        with st.spinner("Calcul de la frontière et simulations Monte Carlo..."):
            # Générer simulations Monte Carlo
            mc_data = monte_carlo_portfolios(
                mu,
                cov_used,
                num_simulations=500,
                min_weight=min_weight,
                max_weight=max_weight,
                allow_short=allow_short,
            )

            # Calculer frontière efficiente (optimisation déterministe)
            frontier_vols, frontier_rets, _ = compute_efficient_frontier(
                mu,
                cov_used,
                num_points=100,
                min_weight=min_weight,
                max_weight=max_weight,
                allow_short=allow_short,
            )

            # Tracer les deux
            fig_frontier = plot_efficient_frontier(
                frontier_vols, frontier_rets, portfolios, mu, cov_used, mc_data
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

            # Statistiques Monte Carlo
            best_mc_sharpe = mc_data["sharpe"].max()
            st.metric(
                "Meilleur Sharpe (MC)",
                f"{best_mc_sharpe:.2f}",
                help="Meilleur ratio de Sharpe trouvé par Monte Carlo",
            )

    # Comparaison des allocations
    st.subheader("Allocations des Portefeuilles")
    fig_weights = plot_weights_comparison(portfolios)
    st.plotly_chart(fig_weights, use_container_width=True)

    # Tableau des statistiques
    st.subheader("Statistiques des Portefeuilles")
    stats_data = []
    for name, weights in portfolios.items():
        ret, vol, sharpe = portfolio_performance(weights, mu, cov_used)
        stats_data.append(
            {
                "Portefeuille": name,
                "Rendement (%)": f"{ret*100:.2f}",
                "Volatilité (%)": f"{vol*100:.2f}",
                "Sharpe Ratio": f"{sharpe:.2f}",
            }
        )

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Performance Historique (Backtest)")

    # Backtests
    backtest_results = {}
    cum_returns_dict = {}

    for name, weights in portfolios.items():
        daily, cum = backtest_constant_weights(weights, returns)
        backtest_results[name] = daily
        cum_returns_dict[name] = cum

    # Graphique de performance cumulée
    st.subheader("Rendements Cumulés")
    fig_perf = plot_cumulative_returns(cum_returns_dict)
    st.plotly_chart(fig_perf, use_container_width=True)

    # Statistiques de backtest
    st.subheader("Statistiques de Backtest")

    backtest_stats_data = []
    for name, daily in backtest_results.items():
        ann_ret, ann_vol, sharpe, mdd = backtest_stats(daily)
        backtest_stats_data.append(
            {
                "Portefeuille": name,
                "Rendement Annualisé (%)": f"{ann_ret*100:.2f}",
                "Volatilité Annualisée (%)": f"{ann_vol*100:.2f}",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown (%)": f"{mdd*100:.2f}",
            }
        )

    backtest_df = pd.DataFrame(backtest_stats_data)
    st.dataframe(backtest_df, use_container_width=True, hide_index=True)

with tab4:
    st.header("Analyse du Risque")

    if show_risk_contrib and portfolios:
        # Sélection du portefeuille pour l'analyse
        selected_portfolio = st.selectbox(
            "Sélectionne un portefeuille pour l'analyse du risque",
            list(portfolios.keys()),
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Allocation du Capital")
            weights_selected = portfolios[selected_portfolio]
            fig_pie_weights = go.Figure(
                data=[
                    go.Pie(
                        labels=weights_selected.index,
                        values=weights_selected.values * 100,
                        hole=0.3,
                        textinfo="label+percent",
                    )
                ]
            )
            fig_pie_weights.update_layout(height=400, title="Poids (%)")
            st.plotly_chart(fig_pie_weights, use_container_width=True)

        with col2:
            st.subheader("Contribution au Risque")
            fig_risk = plot_risk_contribution(weights_selected, cov_used)
            st.plotly_chart(fig_risk, use_container_width=True)

        # Détails des contributions
        st.subheader("Détails de la Décomposition du Risque")

        w = weights_selected.values
        cov_mat = cov_used.values
        port_vol = np.sqrt(w @ cov_mat @ w)
        marginal_contrib = cov_mat @ w
        risk_contrib = w * marginal_contrib / port_vol
        risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

        risk_decomp_df = pd.DataFrame(
            {
                "Actif": weights_selected.index,
                "Poids (%)": weights_selected.values * 100,
                "Volatilité (%)": np.sqrt(np.diag(cov_mat)) * 100,
                "Contribution Marginale": marginal_contrib,
                "Contribution au Risque (%)": risk_contrib_pct,
            }
        )

        st.dataframe(
            risk_decomp_df.style.format(
                {
                    "Poids (%)": "{:.2f}",
                    "Volatilité (%)": "{:.2f}",
                    "Contribution Marginale": "{:.4f}",
                    "Contribution au Risque (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # VaR et CVaR
        st.subheader("Value at Risk (VaR) et Conditional VaR")

        selected_returns = backtest_results[selected_portfolio]

        # VaR à différents niveaux de confiance
        var_95 = np.percentile(selected_returns, 5)
        var_99 = np.percentile(selected_returns, 1)

        # CVaR (Expected Shortfall)
        cvar_95 = selected_returns[selected_returns <= var_95].mean()
        cvar_99 = selected_returns[selected_returns <= var_99].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VaR 95% (quotidien)", f"{var_95*100:.2f}%")
        col2.metric("CVaR 95% (quotidien)", f"{cvar_95*100:.2f}%")
        col3.metric("VaR 99% (quotidien)", f"{var_99*100:.2f}%")
        col4.metric("CVaR 99% (quotidien)", f"{cvar_99*100:.2f}%")

        # Distribution des rendements
        st.subheader("Distribution des Rendements")

        fig_dist = go.Figure()

        fig_dist.add_trace(
            go.Histogram(
                x=selected_returns * 100, name="Rendements", nbinsx=50, opacity=0.7
            )
        )

        fig_dist.add_vline(
            x=var_95 * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="VaR 95%",
            annotation_position="top",
        )

        fig_dist.add_vline(
            x=var_99 * 100,
            line_dash="dash",
            line_color="darkred",
            annotation_text="VaR 99%",
            annotation_position="top",
        )

        fig_dist.update_layout(
            title="Distribution des Rendements Quotidiens",
            xaxis_title="Rendement (%)",
            yaxis_title="Fréquence",
            height=400,
            template="plotly_white",
        )

        st.plotly_chart(fig_dist, use_container_width=True)

# Section bonus: Comparaison et recommandation
st.markdown("---")
st.header("🎯 Recommandation")

if portfolios:
    # Trouver le meilleur portefeuille selon le Sharpe
    best_sharpe = None
    best_sharpe_name = None
    best_sharpe_value = -np.inf

    for name, weights in portfolios.items():
        _, _, sharpe = portfolio_performance(weights, mu, cov_used)
        if sharpe > best_sharpe_value:
            best_sharpe_value = sharpe
            best_sharpe_name = name
            best_sharpe = weights

    st.success(f"🏆 **Meilleur Portefeuille (Sharpe):** {best_sharpe_name}")
    ret, vol, sharpe = portfolio_performance(best_sharpe, mu, cov_used)

    st.markdown(
        f"""
    **Performance Théorique:**
    - 📈 Rendement attendu: **{ret*100:.2f}%** par an
    - 📊 Volatilité: **{vol*100:.2f}%** par an
    - ⚡ Sharpe Ratio: **{sharpe:.2f}**
    """
    )

    # Backtest performance
    if best_sharpe_name in backtest_results:
        daily = backtest_results[best_sharpe_name]
        ann_ret, ann_vol, bt_sharpe, mdd = backtest_stats(daily)

        st.markdown(
            f"""
        **Performance Historique (Backtest):**
        - 📈 Rendement annualisé: **{ann_ret*100:.2f}%**
        - 📊 Volatilité annualisée: **{ann_vol*100:.2f}%**
        - ⚡ Sharpe Ratio: **{bt_sharpe:.2f}**
        - 📉 Max Drawdown: **{mdd*100:.2f}%**
        """
        )

# Footer avec informations supplémentaires
st.markdown("---")
st.markdown(
    """
### 💡 Notes et Recommandations

**Interprétation des Résultats:**
- **Sharpe Ratio > 1**: Bon équilibre risque/rendement
- **Max Drawdown**: Perte maximale historique (plus c'est faible, mieux c'est)
- **VaR/CVaR**: Pertes potentielles dans les pires scénarios

**Limitations:**
- Les performances passées ne garantissent pas les performances futures
- Les corrélations entre actifs peuvent changer dans le temps
- Les modèles supposent des rendements normalement distribués
- Pas de prise en compte des coûts de transaction

**Améliorations Possibles:**
- Backtest out-of-sample (séparation train/test)
- Rebalancement périodique
- Incorporation de contraintes sectorielles
- Optimisation robuste avec scénarios de stress
- Prise en compte des facteurs ESG
"""
)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Développé par Hassan EL QADI | Données fournies par Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True,
)
