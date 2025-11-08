# streamlit_app.py — Subwoofer Array Designer (Web)
# -------------------------------------------------
from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

@dataclass
class Source:
    x: float
    y: float
    delay_s: float = 0.0
    polarity: int = 1   # +1 or -1
    gain: float = 1.0

class AcousticEngine:
    def __init__(self, c: float = 343.0, ref_pressure: float = 2e-5):
        self.c = c
        self.pref = ref_pressure

    def field_complex(self, sources: List[Source], freq: float,
                      X: np.ndarray, Y: np.ndarray, min_r: float = 0.5) -> np.ndarray:
        P = np.zeros_like(X, dtype=np.complex128)
        omega = 2 * np.pi * freq
        for s in sources:
            dx = X - s.x
            dy = Y - s.y
            r = np.sqrt(dx*dx + dy*dy)
            r = np.maximum(r, min_r)
            t = r / self.c + s.delay_s
            phase = omega * t
            contrib = (s.gain * s.polarity / r) * np.exp(1j * phase)
            P += contrib
        return P

    def to_spl(self, P: np.ndarray) -> np.ndarray:
        mag = np.abs(P)
        with np.errstate(divide='ignore'):
            SPL = 20.0 * np.log10(np.maximum(mag / self.pref, 1e-12))
        return SPL

class Arrangements:
    @staticmethod
    def straight(n: int, spacing: float) -> List[Source]:
        xs = (np.arange(n) - (n-1)/2.0) * spacing
        return [Source(x=float(x), y=0.0) for x in xs]

    @staticmethod
    def cardioid(n: int, spacing: float, fb_offset: float, c: float) -> List[Source]:
        sources: List[Source] = []
        pairs = n // 2
        xs = (np.arange(max(2*pairs, n)) - (n-1)/2.0) * spacing
        delay_back = fb_offset / c
        for i in range(pairs):
            x = float(xs[2*i])
            sources.append(Source(x=x, y=0.0, delay_s=0.0, polarity=+1))
            sources.append(Source(x=x, y=-fb_offset, delay_s=delay_back, polarity=-1))
        if n % 2 == 1:
            sources.append(Source(x=float(xs[-1]), y=0.0, delay_s=0.0, polarity=+1))
        return sources

    @staticmethod
    def endfire(n: int, spacing: float, c: float) -> List[Source]:
        delay_step = spacing / c
        y_positions = np.arange(n) * spacing
        sources: List[Source] = []
        for i in range(n):
            sources.append(Source(x=0.0, y=float(y_positions[i]), delay_s=i*delay_step, polarity=+1))
        return sources

    @staticmethod
    def arc_delay(n: int, spacing: float, c: float, radius: float) -> List[Source]:
        xs = (np.arange(n) - (n-1)/2.0) * spacing
        sources: List[Source] = []
        for x in xs:
            delay = (math.sqrt(radius*radius + x*x) - radius) / c
            sources.append(Source(x=float(x), y=0.0, delay_s=delay, polarity=+1))
        return sources

    @staticmethod
    def forty5G(n: int, spacing: float) -> List[Source]:
        idx = np.arange(n) - (n-1)/2.0
        xs = idx * spacing / math.sqrt(2)
        ys = idx * spacing / math.sqrt(2)
        return [Source(x=float(x), y=float(y)) for x, y in zip(xs, ys)]

@st.cache_data(show_spinner=False)
def make_mesh(x_rng: Tuple[float, float], y_rng: Tuple[float, float], res: int):
    x = np.linspace(x_rng[0], x_rng[1], res)
    y = np.linspace(y_rng[0], y_rng[1], res)
    return np.meshgrid(x, y)

def plot_heatmap(X, Y, SPL, sources: List[Source]):
    z = np.clip(SPL, np.nanmax(SPL) - 40.0, np.nanmax(SPL))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=X[0, :], y=Y[:, 0], z=z, colorbar=dict(title='dB')))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), title="Mapa de SPL (dB)")
    fig.add_trace(go.Scatter(x=[s.x for s in sources], y=[s.y for s in sources],
                             mode='markers', name='Subs', marker=dict(size=8, symbol='circle')))
    return fig

st.set_page_config(page_title="Subwoofer Array Designer (Web)", layout="wide")
st.title("Subwoofer Array Designer — Web Demo")

with st.sidebar:
    st.header("Parâmetros")
    arr = st.selectbox("Arranjo (Array A)", ["Straight", "Cardioid", "End-Fire", "Arc Delay", "45G"], index=0)
    nA = st.slider("Nº Subs A", 1, 48, 8)
    spacingA = st.slider("Espaçamento A (m)", 0.1, 6.0, 1.5, 0.1)
    fbA = st.slider("Offset Frente‑Trás A (m)", 0.0, 5.0, 0.8, 0.05)
    radiusA = st.slider("Raio do Arco A (m)", 1.0, 200.0, 30.0, 1.0)

    st.markdown("---")
    useB = st.checkbox("Ativar Array B (secundário)", value=False)
    xB = st.slider("Offset X B (m)", -30.0, 30.0, 6.0, 0.1)
    delayBms = st.slider("Delay extra B (ms)", -50.0, 50.0, 0.0, 0.1)

    st.markdown("---")
    freq = st.slider("Frequência (Hz)", 20.0, 200.0, 63.0, 1.0)
    c = st.slider("Velocidade do som (m/s)", 300.0, 360.0, 343.0, 0.5)

    st.markdown("---")
    grid = st.slider("Tamanho do Grid (m)", 10.0, 200.0, 50.0, 1.0)
    res = st.slider("Resolução do Grid", 80, 400, 220, 10)
    center_audience = st.checkbox("Centralizar Y no público (0→+Y)", value=True)

    st.markdown("---")
    sweep = st.checkbox("Mostrar Varredura (Waterfall)", value=False)
    f0, f1 = st.slider("Faixa de Frequência (Hz)", 20, 200, (40, 120), 1)
    fsteps = st.slider("Passos da varredura", 3, 40, 20, 1)

engine = AcousticEngine(c=float(c))

if arr == 'Straight':
    baseA = Arrangements.straight(nA, spacingA)
elif arr == 'Cardioid':
    baseA = Arrangements.cardioid(nA, spacingA, fb_offset=fbA, c=float(c))
elif arr == 'End-Fire':
    baseA = Arrangements.endfire(nA, spacingA, c=float(c))
elif arr == 'Arc Delay':
    baseA = Arrangements.arc_delay(nA, spacingA, c=float(c), radius=radiusA)
else:
    baseA = Arrangements.forty5G(nA, spacingA)

sources = list(baseA)
if useB:
    doff = delayBms / 1000.0
    sources += [Source(x=s.x + xB, y=s.y, delay_s=s.delay_s + doff, polarity=s.polarity, gain=s.gain) for s in baseA]

if center_audience:
    x_rng = (-grid/2.0, grid/2.0)
    y_rng = (0.0, grid)
else:
    x_rng = (-grid/2.0, grid/2.0)
    y_rng = (-grid/2.0, grid/2.0)

X, Y = make_mesh(x_rng, y_rng, int(res))
P = engine.field_complex(sources, float(freq), X, Y)
SPL = engine.to_spl(P)

fig = plot_heatmap(X, Y, SPL, sources)
st.plotly_chart(fig, use_container_width=True)

import io
csv_buf = io.StringIO()
csv_buf.write('x_m,y_m,SPL_dB\n')
for i in range(SPL.shape[0]):
    for j in range(SPL.shape[1]):
        csv_buf.write(f"{X[i, j]},{Y[i, j]},{SPL[i, j]}\n")
st.download_button("Baixar CSV do mapa", data=csv_buf.getvalue(), file_name="spl_map.csv", mime="text/csv")

if sweep:
    with st.spinner("Calculando varredura..."):
        jmid = X.shape[1] // 2
        freqs = np.linspace(f0, f1, fsteps)
        stack = []
        for f in freqs:
            Pk = engine.field_complex(sources, float(f), X, Y)
            SPLk = engine.to_spl(Pk)
            stack.append(SPLk[:, jmid])
        stack = np.array(stack)
        fig2 = go.Figure()
        fig2.add_trace(go.Heatmap(x=Y[:, 0], y=freqs, z=stack, colorbar=dict(title='dB')))
        fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), title="Varredura: SPL ao longo do Y vs Frequência")
        st.plotly_chart(fig2, use_container_width=True)

st.caption("© Subwoofer Array Designer – Web Demo")
