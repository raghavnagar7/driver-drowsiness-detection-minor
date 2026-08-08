"""Streamlit dashboard for drowsiness detection session logs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT        = Path(__file__).resolve().parent
SESSION_DIR = ROOT / "data" / "sessions"

ALERT_ORDER  = ["OK", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALERT_COLORS = {
    "OK":       "#22c55e",
    "LOW":      "#84cc16",
    "MEDIUM":   "#f59e0b",
    "HIGH":     "#ef4444",
    "CRITICAL": "#7f1d1d",
}
ALERT_DARK = {
    "OK":       "#166534",
    "LOW":      "#3f6212",
    "MEDIUM":   "#92400e",
    "HIGH":     "#991b1b",
    "CRITICAL": "#450a0a",
}

# ── Plotly base theme ────────────────────────────────────────────────────────
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9d1d9", size=12),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(30,35,45,0.7)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        x=0,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1e2330",
        bordercolor="#3d4556",
        font=dict(color="#e6edf3", size=12),
    ),
)


@dataclass
class SessionSummary:
    path: Path
    name: str
    rows: int
    duration_s: float
    estimated_fps: float
    max_score: float
    mean_score: float
    peak_perclos: float
    peak_microsleep: float
    alert_seconds: float
    reliability_pct: float


# ── Page config & CSS ────────────────────────────────────────────────────────

def apply_theme() -> None:
    st.set_page_config(
        page_title="Drowsiness Monitor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Root & body ── */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background: #0d1117;
            color: #c9d1d9;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: #161b22 !important;
            border-right: 1px solid #21262d;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] p {
            color: #8b949e !important;
            font-size: 0.82rem;
        }

        /* ── Metric cards ── */
        div[data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 18px 20px 14px 20px;
            transition: border-color 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #388bfd;
        }
        div[data-testid="stMetric"] label {
            color: #8b949e !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #e6edf3 !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            font-size: 0.78rem !important;
        }

        /* ── Section headers ── */
        h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            border-bottom: 1px solid #21262d;
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #8b949e;
            padding: 10px 22px;
            border-radius: 0;
            font-size: 0.88rem;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            color: #58a6ff !important;
            border-bottom: 2px solid #58a6ff !important;
            background: transparent !important;
        }

        /* ── DataFrames ── */
        .stDataFrame { border-radius: 10px; overflow: hidden; }
        .stDataFrame thead { background: #161b22; }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: #161b22 !important;
            border-radius: 8px !important;
            color: #8b949e !important;
        }

        /* ── Custom components ── */
        .hero-card {
            background: linear-gradient(135deg, #161b22 0%, #1c2333 50%, #1a2236 100%);
            border: 1px solid #21262d;
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }
        .hero-card::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(56,139,253,0.12) 0%, transparent 70%);
            border-radius: 50%;
        }
        .hero-card h1 {
            font-size: 1.80rem;
            font-weight: 700;
            color: #e6edf3;
            margin: 0 0 6px 0;
            letter-spacing: -0.02em;
        }
        .hero-card .subtitle {
            color: #8b949e;
            font-size: 0.88rem;
        }
        .hero-card .badge {
            display: inline-block;
            background: rgba(56,139,253,0.15);
            border: 1px solid rgba(56,139,253,0.3);
            color: #58a6ff;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
            letter-spacing: 0.04em;
        }

        .insight-card {
            background: #161b22;
            border: 1px solid #21262d;
            border-left: 3px solid #388bfd;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 6px 0;
            font-size: 0.88rem;
            color: #c9d1d9;
            line-height: 1.6;
        }
        .insight-card.warn {
            border-left-color: #f59e0b;
        }
        .insight-card.danger {
            border-left-color: #ef4444;
        }
        .insight-card.ok {
            border-left-color: #22c55e;
        }

        .alert-badge {
            display: inline-block;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }

        .section-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: #8b949e;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .chart-card {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .score-ring-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 8px 0;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #484f58; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Data helpers ─────────────────────────────────────────────────────────────

def list_session_files() -> list[Path]:
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("session_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def numeric_column(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


@st.cache_data(show_spinner=False)
def load_session(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    df   = pd.read_csv(path)
    if df.empty:
        return df

    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    for col in ["frame","ear","mar","perclos","blink_rate","score",
                "microsleep_duration","pose_pitch","pose_yaw","pose_roll",
                "pose_score","eyes_reliable"]:
        df[col] = numeric_column(df, col, np.nan if col == "blink_rate" else 0.0)

    if "alert_level" not in df:
        df["alert_level"] = "OK"
    df["alert_level"] = (df["alert_level"].fillna("OK").astype(str)
                         .str.upper().where(lambda s: s.isin(ALERT_ORDER), "OK"))

    df = df.sort_values(["timestamp","frame"], na_position="last").reset_index(drop=True)

    if df["timestamp"].notna().sum() >= 2:
        start = df["timestamp"].dropna().iloc[0]
        df["elapsed_s"] = (df["timestamp"] - start).dt.total_seconds().ffill().fillna(0.0)
    else:
        df["elapsed_s"] = df.index / 30.0
    df["elapsed_s"] = df["elapsed_s"].clip(lower=0)

    fps       = estimate_fps(df)
    w5s       = max(3, int(round(fps * 5)))
    w15s      = max(3, int(round(fps * 15)))

    df["score_smooth"]   = df["score"].rolling(w5s,  min_periods=1).mean()
    df["ear_smooth"]     = df["ear"].rolling(w5s,    min_periods=1).mean()
    df["perclos_pct"]    = df["perclos"] * 100.0
    df["pose_magnitude"] = np.sqrt(df["pose_pitch"]**2 + df["pose_yaw"]**2 + df["pose_roll"]**2)
    df["pose_smooth"]    = df["pose_magnitude"].rolling(w5s, min_periods=1).mean()
    df["attention_load"] = (
        0.55 * df["score"].clip(0,100)
        + 0.25 * df["perclos_pct"].clip(0,100)
        + 0.20 * (df["pose_score"].clip(0,30) / 30.0 * 100)
    ).rolling(w15s, min_periods=1).mean()
    df["risk_band"] = pd.cut(
        df["score"], bins=[-0.1,25,45,70,90,101],
        labels=["OK","LOW","MEDIUM","HIGH","CRITICAL"],
    ).astype(str)
    df["minute"]      = (df["elapsed_s"] // 60).astype(int)
    df["closed_hint"] = (df["perclos"] > 0.20) | (df["microsleep_duration"] > 0.5)
    return df


def estimate_fps(df: pd.DataFrame) -> float:
    if df.empty:
        return 30.0
    dur = float(df.get("elapsed_s", pd.Series([0])).max())
    if dur > 0 and len(df) > 2:
        return max(1.0, min(60.0, (len(df)-1)/dur))
    return 30.0


def summarize(path: Path, df: pd.DataFrame) -> SessionSummary:
    fps  = estimate_fps(df)
    dur  = float(df["elapsed_s"].max()) if not df.empty else 0.0
    a_s  = float((df["alert_level"] != "OK").sum() / fps) if not df.empty else 0.0
    rel  = float(df["eyes_reliable"].mean() * 100) if not df.empty and "eyes_reliable" in df else 100.0
    return SessionSummary(
        path=path, name=path.name, rows=len(df),
        duration_s=dur, estimated_fps=fps,
        max_score=float(df["score"].max()) if not df.empty else 0.0,
        mean_score=float(df["score"].mean()) if not df.empty else 0.0,
        peak_perclos=float(df["perclos_pct"].max()) if not df.empty else 0.0,
        peak_microsleep=float(df["microsleep_duration"].max()) if not df.empty else 0.0,
        alert_seconds=a_s, reliability_pct=rel,
    )


def format_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:   return f"{h}h {m}m {s}s"
    if m:   return f"{m}m {s}s"
    return  f"{s}s"


def alert_episodes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["alert_level","start_s","end_s","duration_s","peak_score"])
    active = df[df["alert_level"] != "OK"].copy()
    if active.empty:
        return pd.DataFrame(columns=["alert_level","start_s","end_s","duration_s","peak_score"])
    group = (active["alert_level"] != active["alert_level"].shift()).cumsum()
    rows  = []
    for _, part in active.groupby(group):
        rows.append({
            "alert_level":  part["alert_level"].iloc[0],
            "start_s":      float(part["elapsed_s"].iloc[0]),
            "end_s":        float(part["elapsed_s"].iloc[-1]),
            "duration_s":   float(part["elapsed_s"].iloc[-1] - part["elapsed_s"].iloc[0]),
            "peak_score":   float(part["score"].max()),
        })
    return pd.DataFrame(rows)


# ── Chart builders ───────────────────────────────────────────────────────────

def _base_fig(height=380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, **_PLOT_LAYOUT)
    return fig


def _add_score_bands(fig: go.Figure) -> None:
    bands = [
        (0,  25,  "rgba(34,197,94,0.04)"),
        (25, 45,  "rgba(132,204,22,0.05)"),
        (45, 70,  "rgba(245,158,11,0.06)"),
        (70, 90,  "rgba(239,68,68,0.07)"),
        (90, 100, "rgba(127,29,29,0.09)"),
    ]
    for y0, y1, color in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")


def risk_timeline(df: pd.DataFrame) -> go.Figure:
    fig = _base_fig(420)
    _add_score_bands(fig)

    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["score"],
        name="Raw score", mode="lines",
        line=dict(color="rgba(56,139,253,0.18)", width=1),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["score_smooth"],
        name="Smoothed score", mode="lines",
        line=dict(color="#58a6ff", width=2.5),
        hovertemplate="<b>%{x:.1f}s</b><br>Score: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["attention_load"],
        name="Fatigue load", mode="lines",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        hovertemplate="<b>%{x:.1f}s</b><br>Fatigue: %{y:.1f}<extra></extra>",
    ))

    # Alert scatter on top.
    alerts = df[df["alert_level"] != "OK"]
    if not alerts.empty:
        fig.add_trace(go.Scatter(
            x=alerts["elapsed_s"], y=alerts["score"],
            name="Alert frames", mode="markers",
            marker=dict(size=7, color=alerts["alert_level"].map(ALERT_COLORS),
                        line=dict(color="#0d1117", width=1)),
            text=alerts["alert_level"],
            hovertemplate="<b>%{text}</b><br>%{x:.1f}s · score %{y:.1f}<extra></extra>",
        ))

    # Threshold lines.
    for val, label, color in [
        (25, "LOW",      "#84cc16"),
        (45, "MEDIUM",   "#f59e0b"),
        (70, "HIGH",     "#ef4444"),
        (90, "CRITICAL", "#7f1d1d"),
    ]:
        fig.add_hline(y=val, line_dash="dash", line_color=color,
                      line_width=1, opacity=0.4,
                      annotation_text=label,
                      annotation_font=dict(size=9, color=color),
                      annotation_position="right")

    fig.update_layout(
        yaxis=dict(title="Risk score", range=[0, 102]),
        xaxis=dict(title="Elapsed (s)"),
    )
    return fig


def eye_chart(df: pd.DataFrame) -> go.Figure:
    fig = _base_fig(340)

    # EAR area.
    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["ear"],
        name="EAR", mode="lines",
        line=dict(color="rgba(88,166,255,0.35)", width=1),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["ear_smooth"],
        name="EAR 5s avg", mode="lines",
        line=dict(color="#58a6ff", width=2.5),
        hovertemplate="<b>%{x:.1f}s</b><br>EAR: %{y:.3f}<extra></extra>",
    ))
    # PERCLOS filled area.
    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["perclos_pct"],
        name="PERCLOS %", mode="lines",
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.12)",
        line=dict(color="#f59e0b", width=1.8),
        yaxis="y2",
        hovertemplate="<b>%{x:.1f}s</b><br>PERCLOS: %{y:.1f}%<extra></extra>",
    ))
    # Microsleep spikes.
    ms = df[df["microsleep_duration"] > 0.3]
    if not ms.empty:
        fig.add_trace(go.Scatter(
            x=ms["elapsed_s"], y=ms["microsleep_duration"],
            name="Microsleep", mode="markers",
            marker=dict(symbol="line-ns", size=14, color="#ef4444",
                        line=dict(width=2, color="#ef4444")),
            yaxis="y2",
            hovertemplate="<b>%{x:.1f}s</b><br>Microsleep: %{y:.2f}s<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(title="Elapsed (s)"),
        yaxis=dict(title="EAR"),
        yaxis2=dict(title="PERCLOS / µsleep", overlaying="y", side="right",
                    tickformat=".0f", ticksuffix="%",
                    gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def pose_chart(df: pd.DataFrame) -> go.Figure:
    fig = _base_fig(340)

    colors = {"Pitch": "#22c55e", "Yaw": "#f59e0b", "Roll": "#ef4444"}
    for col, name in [("pose_pitch","Pitch"),("pose_yaw","Yaw"),("pose_roll","Roll")]:
        fig.add_trace(go.Scatter(
            x=df["elapsed_s"], y=df[col],
            name=name, mode="lines",
            line=dict(color=colors[name], width=1.8),
            hovertemplate=f"<b>%{{x:.1f}}s</b><br>{name}: %{{y:.1f}}°<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=df["elapsed_s"], y=df["pose_score"],
        name="Pose risk", mode="lines",
        line=dict(color="#c9d1d9", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x:.1f}s</b><br>Pose risk: %{y:.1f}<extra></extra>",
    ))

    # Head-down threshold marker.
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=1)

    fig.update_layout(
        xaxis=dict(title="Elapsed (s)"),
        yaxis=dict(title="Degrees"),
        yaxis2=dict(title="Pose risk score", overlaying="y", side="right",
                    range=[0, max(30, df["pose_score"].max() + 2)],
                    gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def minute_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    grouped = df.groupby("minute", as_index=False).agg(
        avg_score      = ("score",               "mean"),
        peak_score     = ("score",               "max"),
        perclos        = ("perclos_pct",          "mean"),
        pose_risk      = ("pose_score",           "mean"),
        peak_microsleep= ("microsleep_duration",  "max"),
    )
    matrix = grouped[["avg_score","peak_score","perclos","pose_risk","peak_microsleep"]].T
    labels = ["Avg Score","Peak Score","PERCLOS (%)","Pose Risk","Peak µsleep (s)"]

    fig = px.imshow(
        matrix,
        labels=dict(x="Minute", y="Signal", color=""),
        x=[f"m{m}" for m in grouped["minute"]],
        y=labels,
        color_continuous_scale=["#0d1117","#1c2333","#1e4620","#f59e0b","#ef4444","#7f1d1d"],
        aspect="auto",
        text_auto=".1f",
    )
    fig.update_traces(textfont=dict(size=10, color="#e6edf3"))
    fig.update_layout(
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=12, b=12),
        font=dict(color="#c9d1d9", size=11),
        coloraxis_showscale=False,
    )
    return fig


def distribution_charts(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    fps = estimate_fps(df)

    # Alert time bar chart.
    alert_counts = (df["alert_level"].value_counts()
                    .reindex(ALERT_ORDER, fill_value=0)
                    .reset_index())
    alert_counts.columns   = ["alert_level", "frames"]
    alert_counts["seconds"] = (alert_counts["frames"] / fps).round(1)

    bar = go.Figure()
    for _, row in alert_counts.iterrows():
        bar.add_trace(go.Bar(
            x=[row["alert_level"]], y=[row["seconds"]],
            name=row["alert_level"],
            marker_color=ALERT_COLORS[row["alert_level"]],
            showlegend=False,
            hovertemplate=f"<b>{row['alert_level']}</b><br>%{{y:.1f}}s<extra></extra>",
        ))
    # Build layout without xaxis/yaxis first, then add them explicitly to avoid
    # duplicate-keyword error when unpacking _PLOT_LAYOUT (which already has those keys).
    _base = {k: v for k, v in _PLOT_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    bar.update_layout(
        height=300, bargap=0.35,
        yaxis=dict(title="Seconds", **_PLOT_LAYOUT["yaxis"]),
        xaxis=dict(**_PLOT_LAYOUT["xaxis"]),
        **_base,
    )
    # Risk donut.
    risk_counts = (df["risk_band"].value_counts()
                   .reindex(ALERT_ORDER, fill_value=0)
                   .reset_index())
    risk_counts.columns = ["risk_band","frames"]

    donut = go.Figure(go.Pie(
        labels=risk_counts["risk_band"],
        values=risk_counts["frames"],
        hole=0.62,
        marker=dict(
            colors=[ALERT_COLORS[r] for r in risk_counts["risk_band"]],
            line=dict(color="#0d1117", width=2),
        ),
        textfont=dict(size=11, color="#e6edf3"),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    # Centre annotation.
    safe_pct = int(round((risk_counts.loc[risk_counts["risk_band"] == "OK", "frames"].sum()
                          / max(risk_counts["frames"].sum(), 1)) * 100))
    donut.add_annotation(
        text=f"<b>{safe_pct}%</b><br><span style='font-size:10px;color:#8b949e'>SAFE</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#e6edf3"),
        align="center",
    )
    donut.update_layout(height=300, showlegend=False, **_PLOT_LAYOUT)
    return bar, donut


def score_gauge(score: float) -> go.Figure:
    """Semi-circle gauge for peak risk score."""
    color = (
        "#22c55e" if score < 25 else
        "#84cc16" if score < 45 else
        "#f59e0b" if score < 70 else
        "#ef4444" if score < 90 else
        "#7f1d1d"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="/100", font=dict(size=28, color="#e6edf3")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="#30363d",
                      tickfont=dict(color="#8b949e", size=10)),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  25], color="rgba(34,197,94,0.06)"),
                dict(range=[25, 45], color="rgba(132,204,22,0.07)"),
                dict(range=[45, 70], color="rgba(245,158,11,0.08)"),
                dict(range=[70, 90], color="rgba(239,68,68,0.09)"),
                dict(range=[90,100], color="rgba(127,29,29,0.10)"),
            ],
        ),
        title=dict(text="Peak Score", font=dict(size=12, color="#8b949e")),
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=0),
        font=dict(color="#c9d1d9"),
    )
    return fig


def blink_rate_chart(df: pd.DataFrame) -> go.Figure:
    br = df[df["blink_rate"].notna() & (df["blink_rate"] > 0)]
    if br.empty:
        return _base_fig(240)

    fig = _base_fig(240)
    fig.add_hrect(y0=8, y1=30, fillcolor="rgba(34,197,94,0.06)", line_width=0,
                  annotation_text="Normal", annotation_font=dict(size=9, color="#22c55e"),
                  annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=br["elapsed_s"], y=br["blink_rate"],
        name="Blinks/min", mode="lines",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="#58a6ff", width=1.8),
        hovertemplate="<b>%{x:.1f}s</b><br>%{y:.0f} blinks/min<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Elapsed (s)"),
        yaxis=dict(title="Blinks/min"),
    )
    return fig


def comparison_chart(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary_df["mean_score"],
        y=summary_df["peak_microsleep"],
        mode="markers+text",
        marker=dict(
            size=summary_df["alert_seconds"].clip(1).apply(lambda v: max(10, min(40, v * 0.8))),
            color=summary_df["max_score"],
            colorscale=[[0,"#22c55e"],[0.4,"#f59e0b"],[0.8,"#ef4444"],[1,"#7f1d1d"]],
            cmin=0, cmax=100,
            line=dict(color="#0d1117", width=1.5),
            showscale=True,
            colorbar=dict(
                title=dict(text="Peak score", font=dict(color="#8b949e", size=10)),
                thickness=12,
                tickfont=dict(color="#8b949e", size=10),
            ),
        ),
        text=summary_df["name"].str.replace("session_","").str.replace(".csv",""),
        textposition="top center",
        textfont=dict(size=9, color="#8b949e"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Mean score: %{x:.1f}<br>"
            "Peak µsleep: %{y:.2f}s<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=420,
        xaxis=dict(title="Average Score", **_PLOT_LAYOUT["xaxis"]),
        yaxis=dict(title="Peak Microsleep (s)", **_PLOT_LAYOUT["yaxis"]),
        **{k: v for k, v in _PLOT_LAYOUT.items() if k not in ("xaxis","yaxis")},
    )
    return fig


# ── Insight generation ───────────────────────────────────────────────────────

def insight_lines(summary: SessionSummary, df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return list of (text, css_class) tuples."""
    out: list[tuple[str, str]] = []

    if not df.empty:
        pk = df.loc[df["score"].idxmax()]
        out.append((
            f"Peak risk was <b>{summary.max_score:.0f}/100</b> at "
            f"{pk['elapsed_s']:.1f}s — PERCLOS {pk['perclos_pct']:.1f}%, "
            f"pose risk {pk['pose_score']:.0f}.",
            "warn" if summary.max_score >= 45 else "",
        ))

    if summary.peak_microsleep >= 1.5:
        out.append((
            f"⚠ Longest eye-closure was <b>{summary.peak_microsleep:.1f}s</b> "
            f"— qualifies as a microsleep event.",
            "danger",
        ))
    elif summary.peak_microsleep > 0:
        out.append((
            f"Eye closures were brief; longest was <b>{summary.peak_microsleep:.1f}s</b>.",
            "ok",
        ))

    if summary.alert_seconds > 10:
        out.append((
            f"Alerts were active for <b>{format_duration(summary.alert_seconds)}</b> "
            f"— consider a rest break.",
            "danger",
        ))
    elif summary.alert_seconds > 0:
        out.append((
            f"Alerts fired for <b>{format_duration(summary.alert_seconds)}</b> total.",
            "warn",
        ))
    else:
        out.append(("No alert episodes recorded. ✓", "ok"))

    if summary.reliability_pct < 90:
        out.append((
            f"Eye tracking reliability was <b>{summary.reliability_pct:.0f}%</b> "
            f"— check camera angle or lighting conditions.",
            "warn",
        ))
    else:
        out.append((
            f"Eye tracking reliability was strong at <b>{summary.reliability_pct:.0f}%</b>.",
            "ok",
        ))

    if not df.empty and df["pose_score"].max() >= 12:
        out.append((
            "Head-pose contributed meaningfully to risk — review the pose chart near alert periods.",
            "warn",
        ))
    return out


# ── Render functions ─────────────────────────────────────────────────────────

def render_session(df: pd.DataFrame, summary: SessionSummary) -> None:

    # Hero card.
    level_color = (
        "#22c55e" if summary.max_score < 25 else
        "#f59e0b" if summary.max_score < 70 else
        "#ef4444"
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>🧠 Driver Drowsiness Analytics</h1>
            <p class="subtitle" style="margin-bottom:12px;">
                <span class="badge">SESSION</span>
                <span class="badge">{format_duration(summary.duration_s)}</span>
                <span class="badge">{summary.rows:,} frames</span>
                <span class="badge">{summary.estimated_fps:.1f} FPS</span>
            </p>
            <p class="subtitle">{summary.name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row.
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Peak Score",      f"{summary.max_score:.0f} / 100")
    c2.metric("Avg Score",       f"{summary.mean_score:.1f}")
    c3.metric("Peak PERCLOS",    f"{summary.peak_perclos:.1f}%")
    c4.metric("Peak µSleep",     f"{summary.peak_microsleep:.2f}s")
    c5.metric("Alert Time",      format_duration(summary.alert_seconds))
    c6.metric("Tracking",        f"{summary.reliability_pct:.0f}%")

    st.divider()

    # ── Row 1: Risk timeline (full width) ──
    st.markdown('<p class="section-label">Risk Timeline</p>', unsafe_allow_html=True)
    st.plotly_chart(risk_timeline(df), use_container_width=True)

    # ── Row 2: Eye + Pose (side by side) ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<p class="section-label">Eye Behaviour (EAR & PERCLOS)</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(eye_chart(df), use_container_width=True)
    with col_r:
        st.markdown('<p class="section-label">Head Pose Angles & Pose Risk</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(pose_chart(df), use_container_width=True)

    # ── Row 3: Gauge + Blink rate ──
    col_g, col_b = st.columns([1, 2])
    with col_g:
        st.markdown('<p class="section-label">Peak Risk Gauge</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(score_gauge(summary.max_score), use_container_width=True)
    with col_b:
        st.markdown('<p class="section-label">Blink Rate Over Session</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(blink_rate_chart(df), use_container_width=True)

    # ── Row 4: Heatmap (full width) ──
    st.markdown('<p class="section-label">Fatigue Load Heatmap (per minute)</p>',
                unsafe_allow_html=True)
    st.plotly_chart(minute_heatmap(df), use_container_width=True)

    # ── Row 5: Alert bar + Donut ──
    col_a, col_d = st.columns(2)
    bar, donut = distribution_charts(df)
    with col_a:
        st.markdown('<p class="section-label">Alert Level Distribution (seconds)</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(bar, use_container_width=True)
    with col_d:
        st.markdown('<p class="section-label">Risk Band Breakdown</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(donut, use_container_width=True)

    st.divider()

    # ── Insights ──
    st.markdown('<p class="section-label">Session Insights</p>', unsafe_allow_html=True)
    insights = insight_lines(summary, df)
    n_cols   = 2
    rows     = [insights[i:i+n_cols] for i in range(0, len(insights), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, (text, cls) in zip(cols, row):
            with col:
                st.markdown(
                    f'<div class="insight-card {cls}">{text}</div>',
                    unsafe_allow_html=True,
                )

    # ── Alert episodes table ──
    episodes = alert_episodes(df)
    if not episodes.empty:
        st.markdown('<p class="section-label" style="margin-top:16px;">Alert Episodes</p>',
                    unsafe_allow_html=True)

        def _fmt_row(row):
            color = ALERT_COLORS.get(row["alert_level"], "#8b949e")
            return (
                f'<span class="alert-badge" '
                f'style="background:{ALERT_DARK.get(row["alert_level"],"#1c2333")};'
                f'color:{color};border:1px solid {color};">'
                f'{row["alert_level"]}</span>'
            )

        episodes_display       = episodes.copy()
        episodes_display["at"] = episodes_display["start_s"].apply(format_duration)
        episodes_display["end"] = episodes_display["end_s"].apply(format_duration)
        episodes_display["dur"] = episodes_display["duration_s"].apply(format_duration)
        episodes_display["peak_score"] = episodes_display["peak_score"].round(1)

        st.dataframe(
            episodes_display[["alert_level","at","end","dur","peak_score"]]
            .rename(columns={"alert_level":"Level","at":"Start",
                              "end":"End","dur":"Duration","peak_score":"Peak Score"}),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("📋 Raw session data"):
        st.dataframe(df, use_container_width=True, height=340)


def render_comparison(paths: list[Path]) -> None:
    if not paths:
        st.info("Select sessions in the sidebar to compare.")
        return

    rows = []
    for path in paths:
        df = load_session(str(path))
        s  = summarize(path, df)
        rows.append({
            "name":            s.name,
            "duration":        format_duration(s.duration_s),
            "fps":             f"{s.estimated_fps:.1f}",
            "max_score":       round(s.max_score, 1),
            "mean_score":      round(s.mean_score, 1),
            "peak_perclos":    f"{s.peak_perclos:.1f}%",
            "peak_microsleep": f"{s.peak_microsleep:.2f}s",
            "alert_seconds":   round(s.alert_seconds, 1),
            "reliability":     f"{s.reliability_pct:.0f}%",
            # For chart
            "alert_seconds_n": s.alert_seconds,
        })

    if not rows:
        return

    summary_df = pd.DataFrame(rows)

    st.markdown('<p class="section-label">Session Comparison Table</p>',
                unsafe_allow_html=True)
    st.dataframe(
        summary_df.drop(columns=["alert_seconds_n"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<p class="section-label" style="margin-top:16px;">Score vs Microsleep Scatter</p>',
                unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "name":            [r["name"] for r in rows],
        "mean_score":      [pd.to_numeric(r["mean_score"]) for r in rows],
        "peak_microsleep": [pd.to_numeric(str(r["peak_microsleep"]).replace("s","")) for r in rows],
        "max_score":       [pd.to_numeric(r["max_score"]) for r in rows],
        "alert_seconds":   [r["alert_seconds_n"] for r in rows],
    })
    st.plotly_chart(comparison_chart(chart_df), use_container_width=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(files: list[Path]):
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:8px 0 16px 0;">
                <p style="font-size:1.1rem;font-weight:700;color:#e6edf3;margin:0;">
                    🧠 Drowsiness Monitor
                </p>
                <p style="font-size:0.75rem;color:#8b949e;margin:4px 0 0 0;">
                    Post-session analytics dashboard
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Review session**")
        selected = st.selectbox(
            "session_select",
            files,
            format_func=lambda p: p.name.replace("session_","").replace(".csv",""),
            label_visibility="collapsed",
        )

        st.markdown("**Compare sessions**", help="Select up to 8 sessions to compare.")
        compare = st.multiselect(
            "session_compare",
            files,
            default=files[:min(4, len(files))],
            format_func=lambda p: p.name.replace("session_","").replace(".csv",""),
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            "<p style='font-size:0.75rem;color:#8b949e;'>"
            "Run a session, quit with <b>Q</b>, then refresh this page."
            "</p>",
            unsafe_allow_html=True,
        )

    return selected, compare


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    apply_theme()

    files = list_session_files()
    if not files:
        st.markdown(
            f"""
            <div class="hero-card">
                <h1>🧠 Drowsiness Monitor</h1>
                <p class="subtitle">No session files found in <code>{SESSION_DIR}</code>.</p>
                <p class="subtitle">Run a detection session first, then come back here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    selected, compare = render_sidebar(files)

    df = load_session(str(selected))
    if df.empty:
        st.warning("Selected session file is empty.")
        return

    summary = summarize(selected, df)

    tab_session, tab_compare = st.tabs(["📊  Session Review", "📈  Compare Runs"])
    with tab_session:
        render_session(df, summary)
    with tab_compare:
        render_comparison(compare)


if __name__ == "__main__":
    main()
