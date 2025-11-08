# streamlit_app.py — Subwoofer Array Designer — NewDevice (v3.3)
from dataclasses import dataclass
import math, os, json, io, base64
from typing import List, Tuple, Optional, Dict
import numpy as np
import streamlit as st
import plotly.graph_objects as go

@dataclass
class Source:
    x: float
    y: float
    delay_s: float = 0.0
    polarity: int = 1
    gain: float = 1.0

class AcousticEngine:
    def __init__(self, c: float = 343.0, ref_pressure: float = 2e-5):
        self.c = c; self.pref = ref_pressure
    def field_complex(self, sources: List[Source], freq: float, X: np.ndarray, Y: np.ndarray, min_r: float = 0.5) -> np.ndarray:
        P = np.zeros_like(X, dtype=np.complex128)
        omega = 2 * np.pi * freq
        for s in sources:
            dx = X - s.x; dy = Y - s.y
            r = np.sqrt(dx*dx + dy*dy); r = np.maximum(r, min_r)
            t = r / self.c + s.delay_s
            P += (s.gain * s.polarity / r) * np.exp(1j * omega * t)
        return P
    def to_spl(self, P: np.ndarray) -> np.ndarray:
        mag = np.abs(P)
        with np.errstate(divide='ignore'):
            return 20.0 * np.log10(np.maximum(mag / self.pref, 1e-12))

class Arrangements:
    @staticmethod
    def straight(n, spacing): xs=(np.arange(n)-(n-1)/2.0)*spacing; return [Source(float(x),0.0) for x in xs]
    @staticmethod
    def cardioid(n, spacing, fb_offset, c):
        out=[]; pairs=n//2; xs=(np.arange(max(2*pairs,n))-(n-1)/2.0)*spacing; delay_back=fb_offset/c
        for i in range(pairs): x=float(xs[2*i]); out+=[Source(x,0.0,0.0,+1), Source(x,-fb_offset,delay_back,-1)]
        if n%2==1: out.append(Source(float(xs[-1]),0.0,0.0,+1)); return out
    @staticmethod
    def endfire(n, spacing, c):
        delay_step=spacing/c; y=np.arange(n)*spacing; return [Source(0.0,float(yv),i*delay_step,+1) for i,yv in enumerate(y)]
    @staticmethod
    def arc_delay(n, spacing, c, radius):
        xs=(np.arange(n)-(n-1)/2.0)*spacing; return [Source(float(x),0.0,(math.sqrt(radius*radius+x*x)-radius)/c,+1) for x in xs]
    @staticmethod
    def forty5G(n, spacing):
        idx=np.arange(n)-(n-1)/2.0; xs=idx*spacing/np.sqrt(2); ys=idx*spacing/np.sqrt(2); return [Source(float(x),float(y)) for x,y in zip(xs,ys)]

# ---- Helpers ----
def human_name(m): return f"{m.get('brand','')} – {m.get('model','')}"

@st.cache_data(show_spinner=False)
def load_models(path="data/models.json"):
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f: data=json.load(f)
            if isinstance(data,list) and data: return data
        except Exception: pass
    return []

@st.cache_data(show_spinner=False)
def make_mesh(x_rng, y_rng, res):
    x=np.linspace(x_rng[0],x_rng[1],res); y=np.linspace(y_rng[0],y_rng[1],res); return np.meshgrid(x,y)

def plot_heatmap(X,Y,SPL,sources):
    z=np.clip(SPL,np.nanmax(SPL)-40.0,np.nanmax(SPL)); fig=go.Figure()
    fig.add_trace(go.Heatmap(x=X[0,:],y=Y[:,0],z=z,colorbar=dict(title='dB'))); fig.update_yaxes(scaleanchor="x",scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10),title="Mapa de SPL (dB)")
    fig.add_trace(go.Scatter(x=[s.x for s in sources],y=[s.y for s in sources],mode='markers',name='Subs',marker=dict(size=8,symbol='circle')))
    return fig

# ---- UI ----
st.set_page_config(page_title="SubArray Designer — NewDevice v3.3", layout="wide")
st.title("Subwoofer Array Designer — NewDevice (v3.3)")

if "catalog" not in st.session_state:
    st.session_state.catalog = load_models()

def save_catalog_download_button(catalog: List[Dict]):
    data = json.dumps(catalog, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("Baixar catálogo (models.json)", data=data, file_name="models.json", mime="application/json")

def upload_catalog_widget():
    up = st.file_uploader("Importar catálogo (models.json)", type=["json"])
    if up is not None:
        try:
            data = json.load(up)
            if isinstance(data, list):
                st.session_state.catalog = data
                st.success(f"Catálogo importado com {len(data)} registros.")
        except Exception as e:
            st.error(f"Erro ao importar: {e}")

catalog = st.session_state.catalog
models = catalog  # alias

with st.sidebar:
    st.header("Catálogo")
    # Filter by Fabricante/Modelo
    brand_filter = st.text_input("Filtrar por Fabricante:", "")
    model_filter = st.text_input("Filtrar por Modelo:", "")
    filtered = []
    for m in models:
        ok = True
        if brand_filter and brand_filter.lower() not in m.get("brand","").lower(): ok=False
        if model_filter and model_filter.lower() not in m.get("model","").lower(): ok=False
        if ok: filtered.append(m)
    show_list = filtered if (brand_filter or model_filter) else models
    name_list=[human_name(m) for m in show_list]

    sel_name=st.selectbox("Modelo real (opcional):",["(sem modelo)"]+name_list, index=0)
    selected=None
    if sel_name!="(sem modelo)":
        idx=name_list.index(sel_name); selected=show_list[idx]
        with st.expander("Especificações do modelo"):
            for k in ["brand","model","drivers","type","directivity"]:
                st.write(f"**{k.capitalize()}:**", selected.get(k))
            fr=selected.get("freq_hz")
            if isinstance(fr,list) and len(fr)==2: st.write("**Faixa útil (Hz):**", f"{fr[0]} – {fr[1]}")
            st.write("**Potência (W):**", selected.get("power_w"))
            st.write("**Sensibilidade (dB):**", selected.get("sensitivity_db"))
            st.write("**SPL máx (dB):**", selected.get("max_spl_db"))
            if selected.get("product_url"): st.markdown(f"[Página do produto]({selected['product_url']})")
            if selected.get("drawing_url"): st.markdown(f"[Desenho / Ficha técnica]({selected['drawing_url']})")
            if selected.get("spec_url"): st.markdown(f"[Fonte de especificação]({selected['spec_url']})")
            st.info(f"Status da ficha: {selected.get('spec_status','pending')}")

    st.markdown("---")
    st.subheader("Catálogo (Admin)")
    with st.form("add_model_form", clear_on_submit=True):
        st.write("Adicionar **Fabricante** e **Modelo** de caixa")
        brand_new = st.text_input("Fabricante*", "")
        model_new = st.text_input("Modelo*", "")
        drivers = st.text_input("Drivers (ex.: 2x18")", "")
        tipo = st.selectbox("Tipo", ["Passive", "Powered", "Powered (band-pass)", "Various"])
        directivity = st.selectbox("Diretividade", ["Omni", "Cardioid", "Various"])
        freq_lo = st.number_input("Freq. mínima (Hz)", min_value=10.0, max_value=200.0, value=40.0, step=1.0)
        freq_hi = st.number_input("Freq. máxima (Hz)", min_value=40.0, max_value=400.0, value=120.0, step=1.0)
        power_w = st.number_input("Potência (W RMS)", min_value=0.0, value=0.0, step=100.0)
        sensitivity_db = st.number_input("Sensibilidade (dB @1W/1m)", min_value=0.0, value=0.0, step=0.5)
        max_spl_db = st.number_input("SPL máx (dB)", min_value=0.0, value=0.0, step=0.5)
        product_url = st.text_input("URL do produto", "")
        drawing_url = st.text_input("URL do desenho/ficha", "")
        spec_url = st.text_input("URL da especificação", "")
        submitted = st.form_submit_button("Adicionar ao catálogo")
        if submitted:
            if not brand_new or not model_new:
                st.warning("Preencha Fabricante e Modelo.")
            else:
                entry = {
                    "brand": brand_new.strip(),
                    "model": model_new.strip(),
                    "drivers": drivers or None,
                    "type": tipo,
                    "directivity": directivity,
                    "freq_hz": [float(freq_lo), float(freq_hi)],
                    "power_w": (None if power_w<=0 else float(power_w)),
                    "sensitivity_db": (None if sensitivity_db<=0 else float(sensitivity_db)),
                    "max_spl_db": (None if max_spl_db<=0 else float(max_spl_db)),
                    "product_url": product_url or None,
                    "drawing_url": drawing_url or None,
                    "spec_url": spec_url or None,
                    "spec_status": "custom"
                }
                st.session_state.catalog.append(entry)
                st.success(f"Adicionado: {entry['brand']} – {entry['model']}")

    # Grouped view by fabricante with counts
    st.write("**Resumo por Fabricante**")
    counts = {}
    for m in st.session_state.catalog:
        counts[m.get("brand","(sem fabricante)")] = counts.get(m.get("brand","(sem fabricante)"), 0) + 1
    for b, n in sorted(counts.items(), key=lambda x: x[0].lower()):
        st.write(f"- {b}: {n} modelo(s)")

    st.markdown("---")
    save_catalog_download_button(st.session_state.catalog)
    upload_catalog_widget()

# ---- Simulação ----
def selected_freq(selected, default=63.0):
    fr=selected.get("freq_hz") if selected else None
    if isinstance(fr,list) and len(fr)==2:
        lo,hi=fr; 
        try: 
            return max(20.0,min(200.0,(float(lo or default)+float(hi or default))/2.0))
        except Exception: 
            return default
    return default

col1, col2 = st.columns([2,1])
with col2:
    st.subheader("Parâmetros do Array A")
    arr=st.selectbox("Arranjo",["Straight","Cardioid","End-Fire","Arc Delay","45G"],index=0)
    nA=st.slider("Nº Subs",1,48,8); spacingA=st.slider("Espaçamento (m)",0.1,6.0,1.5,0.1)
    fbA=st.slider("Offset Frente‑Trás (m)",0.0,5.0,0.8,0.05); radiusA=st.slider("Raio do Arco (m)",1.0,200.0,30.0,1.0)

    st.markdown("---")
    useB=st.checkbox("Ativar Array B (secundário)",value=False)
    xB=st.slider("Offset X B (m)",-30.0,30.0,6.0,0.1); delayBms=st.slider("Delay extra B (ms)",-50.0,50.0,0.0,0.1)

    st.markdown("---")
    freq=st.slider("Frequência (Hz)",20.0,200.0,float(selected_freq(selected if 'selected' in locals() else None)),1.0)
    c=st.slider("Velocidade do som (m/s)",300.0,360.0,343.0,0.5)
    grid=st.slider("Tamanho do Grid (m)",10.0,200.0,50.0,1.0); res=st.slider("Resolução do Grid",80,400,220,10)
    center_audience=st.checkbox("Centralizar Y no público (0→+Y)",True)

with col1:
    engine=AcousticEngine(c=float(c))
    if arr=='Straight': baseA=Arrangements.straight(nA,spacingA)
    elif arr=='Cardioid': baseA=Arrangements.cardioid(nA,spacingA,fb_offset=fbA,c=float(c))
    elif arr=='End-Fire': baseA=Arrangements.endfire(nA,spacingA,c=float(c))
    elif arr=='Arc Delay': baseA=Arrangements.arc_delay(nA,spacingA,c=float(c),radius=radiusA)
    else: baseA=Arrangements.forty5G(nA,spacingA)

    sources=list(baseA)
    if useB:
        doff=delayBms/1000.0; sources += [Source(s.x+xB,s.y,s.delay_s+doff,s.polarity,s.gain) for s in baseA]

    x_rng=(-grid/2.0,grid/2.0); y_rng=(0.0,grid) if center_audience else (-grid/2.0,grid/2.0)
    X,Y=make_mesh(x_rng,y_rng,int(res)); P=engine.field_complex(sources,float(freq),X,Y); SPL=engine.to_spl(P)
    fig=plot_heatmap(X,Y,SPL,sources); st.plotly_chart(fig, use_container_width=True)

    csv_buf = io.StringIO(); csv_buf.write('x_m,y_m,SPL_dB\n')
    for i in range(SPL.shape[0]): 
        for j in range(SPL.shape[1]): csv_buf.write(f"{X[i,j]},{Y[i,j]},{SPL[i,j]}\n")
    st.download_button("Baixar CSV do mapa", data=csv_buf.getvalue(), file_name="spl_map.csv", mime="text/csv")

st.caption("© Subwoofer Array Designer — NewDevice (v3.3 com gestão de Fabricante/Modelo)")
