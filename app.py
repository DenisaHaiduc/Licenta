import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm

# --- 1. FUNCȚIILE MATEMATICE (Imkeller-Tudor & fBm Simulation) ---
def estimeaza_H_Imkeller_Tudor(X):
    if len(X) < 4: return np.nan
    diff2_N = X[2:] - 2 * X[1:-1] + X[:-2]
    V_N = np.sum(diff2_N**2)
    
    X_rar = X[::2]
    if len(X_rar) < 3: return np.nan
    diff2_N2 = X_rar[2:] - 2 * X_rar[1:-1] + X_rar[:-2]
    V_N2 = np.sum(diff2_N2**2)
    
    if V_N == 0 or V_N2 == 0: return np.nan
    return 0.5 * np.log2(V_N2 / V_N) + 0.5

def calculeaza_H_dinamic(preturi, fereastra):
    log_preturi = np.log(preturi.values) 
    H_valori = [np.nan] * len(log_preturi)
    for i in range(fereastra, len(log_preturi)):
        fereastra_curenta = log_preturi[i-fereastra:i]
        H_valori[i] = estimeaza_H_Imkeller_Tudor(fereastra_curenta)
    return H_valori

def simuleaza_fbm(n, H):
    """Simulează o traiectorie fBm de lungime n folosind metoda Cholesky"""
    def R(t, s):
        return 0.5 * (t**(2*H) + s**(2*H) - np.abs(t - s)**(2*H))
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0: cov[i, j] = 0
            else: cov[i, j] = R(i, j)
    cov += 1e-10 * np.eye(n) # stabilitate numerica
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(n)
    return L @ Z

# --- 2. INIȚIALIZARE DASH ---
app = dash.Dash(__name__)

# --- 3. INTERFAȚA (HTML) ---
app.layout = html.Div(className='container', children=[
    
    # MENIUL LATERAL (Stânga)
    html.Div(className='sidebar', style={'width': '25%', 'padding': '20px', 'backgroundColor': '#1A365D', 'overflowY': 'auto'}, children=[
        html.H2("⚙️ Setări Quant"),
        
        html.Label("Criptomonedă:"),
        dcc.Dropdown(
            id='coin-dropdown',
            options=[
                {'label': 'Bitcoin (BTC)', 'value': 'BTC-USD'},
                {'label': 'Ethereum (ETH)', 'value': 'ETH-USD'},
                {'label': 'Solana (SOL)', 'value': 'SOL-USD'}
            ],
            value='BTC-USD',
            style={'color': 'black', 'marginBottom': '20px'} 
        ),
        
        html.Label("Data de început (YYYY-MM-DD):"),
        dcc.Input(id='start-date-input', type='text', value=(datetime.today() - timedelta(days=1500)).strftime('%Y-%m-%d'), style={'width': '95%', 'marginBottom': '10px'}),
        
        html.Label("Data de sfârșit (YYYY-MM-DD):"),
        dcc.Input(id='end-date-input', type='text', value=datetime.today().strftime('%Y-%m-%d'), style={'width': '95%', 'marginBottom': '20px'}),
        
        html.Label("Fereastră calcul Hurst (zile):"),
        dcc.Slider(id='window-slider', min=30, max=300, step=10, value=100, marks={30: '30', 100: '100', 200: '200'}),
        
        html.Br(),
        html.H3("🔮 Setări Prognoză"),
        html.Label("Zile de prognoză în viitor:"),
        dcc.Slider(id='forecast-days', min=7, max=60, step=1, value=30, marks={7: '7', 30: '30', 60: '60'}),
        
        html.Br(), html.Br(),
        html.P("Platformă realizată pentru analiza memoriei pieței, risc și prognoză stochastică.", style={'color': '#A0AEC0', 'fontSize': '30px'})
    ]),
    
    # CONȚINUTUL PRINCIPAL (Dreapta)
    html.Div(className='main-content', style={'width': '75%', 'padding': '30px', 'overflowY': 'auto'}, children=[
        html.H1(id='main-title', children="Analiza Memoriei Pieței"),
        
        # 3 CASETE CU REZULTATE
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
            html.Div(className='metric-box', style={'width': '30%'}, children=[
                html.Div("Preț Curent", style={'color': 'white'}),
                html.Div(id='current-price', className='metric-value')
            ]),
            html.Div(className='metric-box', style={'width': '30%'}, children=[
                html.Div("Hurst Actual", style={'color': 'white'}),
                html.Div(id='current-hurst', className='metric-value')
            ]),
            html.Div(className='metric-box', style={'width': '30%'}, children=[
                html.Div("Regim Piață", style={'color': 'white'}),
                html.Div(id='market-regime', className='metric-value', style={'fontSize': '22px'})
            ])
        ]),
        
        # GRAFICUL 1: PREȚ + HURST + SEMNALE 
        dcc.Graph(id='main-graph', style={'height': '800px', 'marginBottom': '50px'}),
        
        # GRAFICUL 3 (NOU!): PREDICȚIA MONTE CARLO
        html.H2("🔮 Prognoză Stochastică: Monte Carlo fBm", style={'color': '#FFD700'}),
        html.P("Simulăm 100 de traiectorii viitoare bazate pe volatilitatea actuală și parametrul Hurst (memoria pieței).", style={'color': 'white'}),
        dcc.Graph(id='monte-carlo-graph', style={'height': '600px', 'marginBottom': '50px'}),
        
        # GRAFICUL 2: DISTRIBUȚIA RISCULUI 
        html.H2("Analiza Riscului: Distribuția Randamentelor", style={'color': '#FFD700'}),
        dcc.Graph(id='risk-graph', style={'height': '500px'})
    ])
])

# --- 4. LOGICA (Callbacks) ---
@app.callback(
    [Output('main-graph', 'figure'),
     Output('monte-carlo-graph', 'figure'),
     Output('risk-graph', 'figure'),
     Output('current-price', 'children'),
     Output('current-hurst', 'children'),
     Output('market-regime', 'children'),
     Output('market-regime', 'style'),
     Output('main-title', 'children')],
    [Input('coin-dropdown', 'value'),
     Input('start-date-input', 'value'),
     Input('end-date-input', 'value'),
     Input('window-slider', 'value'),
     Input('forecast-days', 'value')]
)
def update_dashboard(coin, start_date, end_date, window, forecast_days):
    # Descarcă datele
    df = yf.download(coin, start=start_date, end=end_date)
    if df.empty:
        return go.Figure(), go.Figure(), go.Figure(), "N/A", "N/A", "N/A", {}, "Date indisponibile"
    
    preturi = df['Close'][coin].dropna() if isinstance(df.columns, pd.MultiIndex) else df['Close'].dropna()

    # 1. Calcul H dinamic
    H_dinamic = calculeaza_H_dinamic(preturi, window)
    
    ultimul_pret = f"${preturi.iloc[-1]:,.2f}"
    val_H = H_dinamic[-1]
    
    if np.isnan(val_H):
        ultimul_H, regime_text, regime_color = "Calcul...", "Se încarcă", "gray"
        val_H = 0.5 # Default pentru simulare daca nu avem destule date
    else:
        ultimul_H = f"{val_H:.3f}"
        if val_H > 0.6: regime_text, regime_color = "Trend Puternic", "#00FF00"
        elif val_H < 0.4: regime_text, regime_color = "Reversie (Haos)", "#FF4136"
        else: regime_text, regime_color = "Zgomot Alb", "#FFD700"

    # Semnale de trading
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for i in range(1, len(H_dinamic)):
        if np.isnan(H_dinamic[i]) or np.isnan(H_dinamic[i-1]): continue
        if H_dinamic[i-1] < 0.6 and H_dinamic[i] >= 0.6:
            buy_x.append(preturi.index[i]), buy_y.append(preturi.values[i])
        elif H_dinamic[i-1] >= 0.5 and H_dinamic[i] < 0.5:
            sell_x.append(preturi.index[i]), sell_y.append(preturi.values[i])

    # GRAFIC 1 (Principal)
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    fig1.add_trace(go.Scatter(x=preturi.index, y=preturi.values, mode='lines', name='Preț', line=dict(color='#FFD700')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='Buy', marker=dict(color='#00FF00', size=12, symbol='triangle-up')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='Sell', marker=dict(color='#FF4136', size=12, symbol='triangle-down')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=preturi.index, y=H_dinamic, mode='lines', name='Hurst', line=dict(color='#00BFFF')), row=2, col=1)
    fig1.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)
    fig1.add_hline(y=0.6, line_dash="dot", line_color="#00FF00", row=2, col=1)
    fig1.update_layout(height=800, paper_bgcolor='#0B192C', plot_bgcolor='#0B192C', font=dict(color='white'), showlegend=False)
    fig1.update_xaxes(gridcolor='#1A365D')
    fig1.update_yaxes(gridcolor='#1A365D')

    # GRAFIC 3 (PROGNOZA MONTE CARLO) - Il punem al doilea in output
    randamente = preturi.pct_change().dropna()
    mu = randamente.mean()
    sigma = randamente.std()
    S0 = preturi.iloc[-1]
    last_date = preturi.index[-1]
    
    viitor_date = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    fig_mc = go.Figure()
    # Afisam o portiune din trecut pentru context
    context_days = 60
    fig_mc.add_trace(go.Scatter(x=preturi.index[-context_days:], y=preturi.values[-context_days:], mode='lines', name='Istoric', line=dict(color='white', width=2)))
    
    toate_traiectoriile = []
    num_sims = 100
    np.random.seed(42) # Pentru rezultate consistente la refresh
    
    for _ in range(num_sims):
        # Generam fBm
        B_H = simuleaza_fbm(forecast_days, val_H)
        # Formula GfBm
        time_steps = np.arange(1, forecast_days + 1)
        # Drift-ul si Volatilitatea scalate in timp
        path = S0 * np.exp((mu - 0.5 * sigma**2) * time_steps + sigma * B_H)
        toate_traiectoriile.append(path)
        # Desenam liniile subtiri
        fig_mc.add_trace(go.Scatter(x=viitor_date, y=path, mode='lines', line=dict(color='#00BFFF', width=1), opacity=0.1, showlegend=False))
        
    # Calculam P5, P50 (Mediana), P95 pentru vizualizare
    traiectorii_matrice = np.array(toate_traiectoriile)
    p5 = np.percentile(traiectorii_matrice, 5, axis=0)
    p50 = np.percentile(traiectorii_matrice, 50, axis=0)
    p95 = np.percentile(traiectorii_matrice, 95, axis=0)
    
    fig_mc.add_trace(go.Scatter(x=viitor_date, y=p95, mode='lines', line=dict(color='red', width=1, dash='dash'), name='Limita Superioara (95%)'))
    fig_mc.add_trace(go.Scatter(x=viitor_date, y=p5, mode='lines', line=dict(color='red', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='Limita Inferioara (5%)'))
    fig_mc.add_trace(go.Scatter(x=viitor_date, y=p50, mode='lines', line=dict(color='#FFD700', width=3), name='Traiectoria Mediana (Asteptata)'))

    fig_mc.update_layout(height=600, paper_bgcolor='#0B192C', plot_bgcolor='#0B192C', font=dict(color='white'), margin=dict(l=20, r=20, t=30, b=20))
    fig_mc.update_xaxes(gridcolor='#1A365D')
    fig_mc.update_yaxes(gridcolor='#1A365D')

    # GRAFIC 2 (RISC FAT TAILS)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=randamente, histnorm='probability density', name='Realitate (Piață)', marker_color='#00BFFF', opacity=0.7))
    mu_norm, std_norm = norm.fit(randamente)
    x_val = np.linspace(randamente.min(), randamente.max(), 100)
    y_val = norm.pdf(x_val, mu_norm, std_norm)
    fig2.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines', name='Teorie (Curba Gauss)', line=dict(color='red', width=3)))
    fig2.update_layout(height=500, paper_bgcolor='#0B192C', plot_bgcolor='#0B192C', font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20))
    fig2.update_xaxes(title="Randament Zilnic", gridcolor='#1A365D')
    fig2.update_yaxes(title="Densitate", gridcolor='#1A365D')

    titlu = f"Dashboard Quant: {coin}"
    stil_regim = {'color': regime_color, 'fontWeight': 'bold', 'fontSize': '22px'}
    
    return fig1, fig_mc, fig2, ultimul_pret, ultimul_H, regime_text, stil_regim, titlu

if __name__ == '__main__':
    app.run(debug=True)