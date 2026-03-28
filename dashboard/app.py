# dashboard/app.py — Streamlit real-time smart manufacturing dashboard

import sys
import os
import time
import threading
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import SIM_TIME_STEP, FAILURE_THRESHOLD, OVERLOAD_THRESHOLD
from simulation.factory import FactorySimulation

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Manufacturing Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .status-running   { background:#d4edda; color:#155724; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:500; }
  .status-overloaded{ background:#fff3cd; color:#856404; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:500; }
  .status-critical  { background:#f8d7da; color:#721c24; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:500; }
  .alert-warning    { background:#fff3cd; color:#856404; border-left:4px solid #ffc107; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px; }
  .alert-critical   { background:#f8d7da; color:#721c24; border-left:4px solid #dc3545; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px; }
  .metric-card      { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────

def _init_session():
    if "sim" not in st.session_state:
        predictor_fn = None
        alert_fn     = None

        try:
            from simulation.predictor import predict_failure_probability
            predictor_fn = predict_failure_probability
        except FileNotFoundError:
            st.warning("Model not found — using heuristic predictor. Run `main.py` to train.")

        if st.session_state.get("llm_enabled", True):
            try:
                from llm.alert_generator import generate_alert_string
                alert_fn = generate_alert_string
            except Exception:
                pass

        st.session_state["sim"] = FactorySimulation(
            predictor_fn=predictor_fn,
            alert_fn=alert_fn,
            time_step=SIM_TIME_STEP,
            speed_factor=st.session_state.get("speed_factor", 1.0),
        )
        st.session_state["sim"].start()
        st.session_state["alerts"] = []

_init_session()
sim: FactorySimulation = st.session_state["sim"]


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")

    machine_sel = st.selectbox("Machine", ["All", "A", "B", "C"])

    speed = st.select_slider(
        "Simulation speed",
        options=[1, 2, 5],
        value=int(st.session_state.get("speed_factor", 1)),
    )
    if speed != st.session_state.get("speed_factor", 1):
        st.session_state["speed_factor"] = speed
        sim.set_speed(float(speed))

    llm_enabled = st.toggle("Enable LLM alerts", value=st.session_state.get("llm_enabled", True))
    st.session_state["llm_enabled"] = llm_enabled

    st.markdown("---")
    st.markdown("**MLflow UI**")
    st.code("mlflow ui\n# → http://localhost:5000", language="bash")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("## 🏭 Smart Manufacturing Digital Twin")
st.markdown("Real-time predictive maintenance powered by ML + SimPy + Gemini Flash 2.0")
st.markdown("---")


# ── Helper: status badge HTML ──────────────────────────────────────────────────

def status_badge(status: str) -> str:
    cls = {
        "Running":    "status-running",
        "Overloaded": "status-overloaded",
        "Critical":   "status-critical",
    }.get(status, "status-running")
    return f'<span class="{cls}">{status}</span>'


# ── Helper: failure probability gauge ─────────────────────────────────────────

def prob_gauge(prob: float, machine_id: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": f"Machine {machine_id} — Failure %", "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#343a40"},
            "steps": [
                {"range": [0, 40],  "color": "#d4edda"},
                {"range": [40, 80], "color": "#fff3cd"},
                {"range": [80, 100],"color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 80,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── Helper: line chart from machine history ────────────────────────────────────

def line_chart(history: list[dict], y_key: str, title: str, color: str) -> go.Figure:
    if not history:
        return go.Figure()
    df  = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[y_key],
        mode="lines", line=dict(color=color, width=1.5),
        name=title,
    ))
    fig.update_layout(
        title=title, height=180,
        margin=dict(t=30, b=20, l=40, r=10),
        xaxis=dict(showticklabels=False),
        yaxis_title=None,
    )
    return fig


# ── MLflow model comparison chart ─────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_mlflow_metrics() -> pd.DataFrame | None:
    try:
        import mlflow
        from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return None
        runs = client.search_runs(exp.experiment_id)
        rows = []
        for r in runs:
            rows.append({
                "Model":     r.data.tags.get("mlflow.runName", r.info.run_id),
                "Accuracy":  r.data.metrics.get("accuracy",  0),
                "Precision": r.data.metrics.get("precision", 0),
                "Recall":    r.data.metrics.get("recall",    0),
                "F1":        r.data.metrics.get("f1",        0),
            })
        return pd.DataFrame(rows)
    except Exception:
        return None


def mlflow_bar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig = go.Figure()
    for m, c in zip(metrics, colors):
        fig.add_trace(go.Bar(name=m, x=df["Model"], y=df[m], marker_color=c))
    fig.update_layout(
        barmode="group", height=300,
        title="Model comparison (MLflow)",
        margin=dict(t=40, b=20, l=40, r=10),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ── Main render loop ───────────────────────────────────────────────────────────

placeholder    = st.empty()
render_count   = 0                  # increments each loop to keep keys unique

while True:
    snapshots = sim.get_all_snapshots()
    machines  = (
        list(snapshots.keys()) if machine_sel == "All" else [machine_sel]
    )

    with placeholder.container():
        # ── Machine cards row ──────────────────────────────────────────────────
        cols = st.columns(len(machines))
        for col, mid in zip(cols, machines):
            s = snapshots[mid]
            with col:
                st.markdown(
                    f"**Machine {mid}** &nbsp; {status_badge(s['status'])}",
                    unsafe_allow_html=True,
                )
                st.metric("Tool wear (min)", s["tool_wear"])
                st.metric("Torque (Nm)",     round(s["torque"], 1))
                st.metric("Speed (rpm)",     round(s["rotational_speed"], 0))
                st.plotly_chart(
                    prob_gauge(s["failure_probability"], mid),
                    use_container_width=True, key=f"gauge_{mid}_{render_count}",
                )

        # ── Real-time sensor charts ────────────────────────────────────────────
        st.markdown("#### Sensor trends")
        for mid in machines:
            m       = sim.get_machine(mid)
            history = list(m.history)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(
                    line_chart(history, "air_temperature", f"M{mid} — Air temp (K)", "#4e79a7"),
                    use_container_width=True, key=f"temp_{mid}_{render_count}",
                )
            with c2:
                st.plotly_chart(
                    line_chart(history, "torque", f"M{mid} — Torque (Nm)", "#f28e2b"),
                    use_container_width=True, key=f"torque_{mid}_{render_count}",
                )
            with c3:
                st.plotly_chart(
                    line_chart(history, "tool_wear", f"M{mid} — Tool wear (min)", "#e15759"),
                    use_container_width=True, key=f"wear_{mid}_{render_count}",
                )

        # ── LLM Alert panel ────────────────────────────────────────────────────
        st.markdown("#### Active alerts")
        any_alert = False
        for mid in machines:
            s = snapshots[mid]
            if s["alert"]:
                any_alert = True
                level     = "CRITICAL" if "CRITICAL" in s["alert"].upper() else "WARNING"
                css_cls   = "alert-critical" if level == "CRITICAL" else "alert-warning"
                st.markdown(
                    f'<div class="{css_cls}">{s["alert"]}</div>',
                    unsafe_allow_html=True,
                )
                if s["alert"] not in st.session_state["alerts"][-10:]:
                    st.session_state["alerts"].append(s["alert"])

        if not any_alert:
            st.success("All monitored machines are operating normally.")

        # ── MLflow comparison ──────────────────────────────────────────────────
        st.markdown("#### Model performance (MLflow)")
        mlf_df = load_mlflow_metrics()
        if mlf_df is not None and not mlf_df.empty:
            st.plotly_chart(
                mlflow_bar_chart(mlf_df),
                use_container_width=True, key=f"mlflow_chart_{render_count}",
            )
        else:
            st.info("No MLflow runs found yet. Run `python main.py` to train models.")

    render_count += 1               # must be outside the `with` block
    time.sleep(SIM_TIME_STEP * 2)
