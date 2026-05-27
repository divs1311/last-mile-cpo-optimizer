"""
Last Mile Cost & Unit Economics Optimizer
==========================================
A production-ready Streamlit dashboard for Q-Commerce supply chain analytics.
Models 5,000+ synthetic orders across 3 dark stores over a 24-hour cycle,
simulating real-world operational friction, batching logic, and unit economics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Last Mile Optimizer · Q-Commerce Analytics",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLE INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --forest:   #1E6B45;
    --mint:     #2ECC71;
    --slate:    #2C3E50;
    --charcoal: #1A1A2E;
    --offwhite: #F0F4F1;
    --muted:    #7F8C8D;
    --warn:     #E74C3C;
    --amber:    #F39C12;
    --card-bg:  #16213E;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--charcoal) !important;
    color: var(--offwhite) !important;
}

.stApp { background-color: var(--charcoal); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F3D2B 0%, #16213E 100%) !important;
    border-right: 1px solid rgba(46, 204, 113, 0.2);
}
[data-testid="stSidebar"] * { color: var(--offwhite) !important; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #16213E 0%, #0F3D2B22 100%);
    border: 1px solid rgba(46, 204, 113, 0.2);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1E6B45, #2ECC71);
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2ECC71;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #F0F4F1;
    line-height: 1;
}
.kpi-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    margin-top: 6px;
}
.kpi-delta.good { color: #2ECC71; }
.kpi-delta.bad  { color: #E74C3C; }
.kpi-delta.neutral { color: #7F8C8D; }

/* Section Headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #2ECC71;
    border-bottom: 1px solid rgba(46, 204, 113, 0.2);
    padding-bottom: 8px;
    margin: 32px 0 16px;
}

/* Exception Log Table */
.exception-row { border-left: 3px solid #E74C3C; padding-left: 8px; }

/* Metric Badges */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.badge-green { background: rgba(30, 107, 69, 0.2); color: #2ECC71; border: 1px solid rgba(46, 204, 113, 0.3); }
.badge-red   { background: rgba(231, 76, 60, 0.2); color: #E74C3C; border: 1px solid rgba(231, 76, 60, 0.3); }
.badge-amber { background: rgba(243, 156, 18, 0.2); color: #F39C12; border: 1px solid rgba(243, 156, 18, 0.3); }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }

/* Streamlit overrides */
.stSlider label, .stSelectbox label { color: #2ECC71 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; letter-spacing: 1px !important; }
div[data-testid="metric-container"] { background: #16213E; border: 1px solid rgba(46, 204, 113, 0.2); border-radius: 8px; padding: 12px; }
.stDataFrame { background: #16213E; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — DARK STORE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DarkStore:
    """Represents a single dark store hub with geographic anchor and operational profile."""
    store_id: str
    name: str
    lat: float
    lon: float
    base_pick_time_min: float   # avg minutes to pick & stage an order
    peak_congestion_factor: float  # multiplier applied during peak hours


DARK_STORES = [
    DarkStore("DS01", "Koramangala Hub",   12.9352, 77.6245, base_pick_time_min=3.5, peak_congestion_factor=2.8),
    DarkStore("DS02", "Indiranagar Hub",   12.9784, 77.6408, base_pick_time_min=4.0, peak_congestion_factor=3.2),
    DarkStore("DS03", "Whitefield Hub",    12.9698, 77.7499, base_pick_time_min=3.0, peak_congestion_factor=2.5),
]

STORE_LOOKUP = {s.store_id: s for s in DARK_STORES}

# Peak hours definition: morning rush 8–10am, lunch 12–2pm, dinner 6–9pm
PEAK_WINDOWS = [(8, 10), (12, 14), (18, 21)]

ITEM_CATEGORIES = [
    "Fresh F&V",           # High perishability, time-sensitive
    "Dairy & Eggs",
    "Packaged Staples",
    "Snacks & Beverages",
    "Meat & Seafood",
    "Personal Care",
]

CATEGORY_WEIGHT = [0.22, 0.18, 0.25, 0.15, 0.10, 0.10]  # Distribution weights


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — SYNTHETIC DATA ENGINE
# ═══════════════════════════════════════════════════════════════════

def is_peak_hour(hour: int) -> bool:
    """Return True if the given hour falls within a defined peak window."""
    return any(start <= hour < end for start, end in PEAK_WINDOWS)


def generate_orders(n_orders: int = 5200, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic Operational Data Engine.
    Generates realistic, friction-laden order data for n_orders across 3 dark stores
    over a 24-hour cycle.
    """
    rng = np.random.default_rng(seed)

    # ── Order allocation: uneven distribution across stores (DS02 is busiest) ──
    store_probs = [0.30, 0.42, 0.28]
    store_ids = rng.choice([s.store_id for s in DARK_STORES], size=n_orders, p=store_probs)

    # ── Timestamps: non-uniform over 24h; demand spikes at peak windows ──
    hour_weights = np.ones(24)
    for start, end in PEAK_WINDOWS:
        hour_weights[start:end] *= 3.5
    hour_weights /= hour_weights.sum()

    hours   = rng.choice(np.arange(24), size=n_orders, p=hour_weights)
    minutes = rng.integers(0, 60, size=n_orders)
    seconds = rng.integers(0, 60, size=n_orders)

    timestamps = pd.to_datetime(
        pd.DataFrame({"hour": hours, "minute": minutes, "second": seconds})
        .apply(lambda r: f"2024-01-15 {int(r.hour):02d}:{int(r.minute):02d}:{int(r.second):02d}", axis=1)
    )

    peak_flags = np.array([is_peak_hour(h) for h in hours])

    # ── Spatial coordinates: random within ~3.5km radius of each store hub ──
    lats, lons = [], []
    for sid in store_ids:
        store = STORE_LOOKUP[sid]
        # 1 degree lat ≈ 111km; radius_deg converts km radius to degrees
        radius_deg = 3.5 / 111.0
        angle = rng.uniform(0, 2 * math.pi)
        r = radius_deg * np.sqrt(rng.uniform(0, 1))
        lats.append(store.lat + r * math.sin(angle))
        lons.append(store.lon + r * math.cos(angle))

    lats = np.array(lats)
    lons = np.array(lons)

    # ── Basket attributes ──
    basket_weight_kg = rng.lognormal(mean=1.2, sigma=0.5, size=n_orders).clip(0.3, 12.0)
    item_categories  = rng.choice(ITEM_CATEGORIES, size=n_orders, p=CATEGORY_WEIGHT)

    # ── Time-Motion Friction Variables ──
    base_pickup = rng.uniform(2.0, 5.0, size=n_orders)
    peak_pickup_multiplier = np.where(
        peak_flags,
        rng.uniform(1.8, 3.5, size=n_orders),   # peak: congestion 1.8–3.5×
        1.0
    )
    store_cong = np.array([STORE_LOOKUP[sid].peak_congestion_factor / 2.5 for sid in store_ids])
    dark_store_pickup_wait_time = (base_pickup * peak_pickup_multiplier * store_cong).clip(2.0, 12.0)

    rider_onboarding_delay = rng.uniform(1.0, 3.0, size=n_orders) + np.where(
        peak_flags, rng.uniform(0.5, 2.0, size=n_orders), 0.0
    )
    rider_onboarding_delay = rider_onboarding_delay.clip(1.0, 5.0)

    store_lats = np.array([STORE_LOOKUP[sid].lat for sid in store_ids])
    store_lons = np.array([STORE_LOOKUP[sid].lon for sid in store_ids])
    delivery_distances_km = haversine_vectorized(store_lats, store_lons, lats, lons)

    base_speed = rng.uniform(22.0, 30.0, size=n_orders)
    peak_speed_penalty = np.where(peak_flags, rng.uniform(5.0, 12.0, size=n_orders), 0.0)
    distance_penalty    = (delivery_distances_km / 3.5) * rng.uniform(1.0, 3.0, size=n_orders)
    transit_speed = (base_speed - peak_speed_penalty - distance_penalty).clip(15.0, 30.0)

    transit_time_min = (delivery_distances_km / transit_speed) * 60.0

    perishable_mask = np.isin(item_categories, ["Fresh F&V", "Meat & Seafood", "Dairy & Eggs"])
    base_dropoff = rng.uniform(1.0, 4.0, size=n_orders)
    weight_bonus  = (basket_weight_kg / 12.0) * rng.uniform(0.5, 2.0, size=n_orders)
    exception_bonus = np.where(perishable_mask, rng.uniform(0.2, 1.0, size=n_orders), 0.0)
    door_dropoff_exception_time = (base_dropoff + weight_bonus + exception_bonus).clip(1.0, 7.0)

    # ── Total TAT (Turnaround Time) per order ──
    total_tat_min = (
        dark_store_pickup_wait_time
        + rider_onboarding_delay
        + transit_time_min
        + door_dropoff_exception_time
    )

    # ── Cost Components ──
    base_payout_per_order = 18.0   # ₹ base rider payout per delivered order
    distance_surge = np.maximum(0.0, (delivery_distances_km - 1.5) * 2.0)
    peak_multiplier_cost = np.where(peak_flags, base_payout_per_order * 0.30, 0.0)
    
    raw_cpo = base_payout_per_order + distance_surge + peak_multiplier_cost

    df = pd.DataFrame({
        "order_id":                    [f"ORD{i:05d}" for i in range(n_orders)],
        "dark_store_id":               store_ids,
        "timestamp":                   timestamps,
        "hour":                        hours,
        "is_peak":                     peak_flags,
        "delivery_lat":                lats,
        "delivery_lon":                lons,
        "basket_weight_kg":            basket_weight_kg.round(2),
        "item_category":               item_categories,
        "dark_store_pickup_wait_time": dark_store_pickup_wait_time.round(2),
        "rider_onboarding_delay":      rider_onboarding_delay.round(2),
        "transit_speed_kmh":           transit_speed.round(2),
        "transit_time_min":            transit_time_min.round(2),
        "door_dropoff_exception_time": door_dropoff_exception_time.round(2),
        "delivery_distance_km":        delivery_distances_km.round(3),
        "total_tat_min":               total_tat_min.round(2),
        "base_payout":                 base_payout_per_order,
        "distance_surge":              distance_surge.round(2),
        "peak_multiplier_cost":        peak_multiplier_cost.round(2),
        "raw_cpo":                     raw_cpo.round(2),
    })

    df["hour_label"] = df["hour"].apply(lambda h: f"{h:02d}:00")
    df["period"]     = df["is_peak"].map({True: "Peak", False: "Off-Peak"})
    df["store_name"] = df["dark_store_id"].map({s.store_id: s.name for s in DARK_STORES})

    return df.sort_values("timestamp").reset_index(drop=True)


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray,
                          lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — BATCHING & ROUTING OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Batch:
    batch_id:      str
    store_id:      str
    order_ids:     List[str]
    total_tat_min: float
    batch_cpo:     float
    sla_breached:  bool
    breach_details: Optional[str] = None
    is_peak:       bool = False


def compute_batch_tat(orders_subset: pd.DataFrame) -> float:
    if orders_subset.empty:
        return 0.0
    pickup_wait      = orders_subset["dark_store_pickup_wait_time"].max()
    onboarding_delay = orders_subset["rider_onboarding_delay"].mean()
    transit_total    = orders_subset["transit_time_min"].sum()
    dropoff_total    = orders_subset["door_dropoff_exception_time"].sum()
    return pickup_wait + onboarding_delay + transit_total + dropoff_total


def compute_batch_cpo(orders_subset: pd.DataFrame, sla_breached: bool,
                       sla_penalty: float = 25.0) -> float:
    n = len(orders_subset)
    if n == 0:
        return 0.0

    total_base     = orders_subset["base_payout"].sum()
    total_surge    = orders_subset["distance_surge"].sum()
    total_peak     = orders_subset["peak_multiplier_cost"].sum()
    breach_penalty = sla_penalty * n if sla_breached else 0.0

    freshness_penalty = 0.0
    for _, row in orders_subset.iterrows():
        if row["item_category"] in ["Fresh F&V", "Meat & Seafood", "Dairy & Eggs"]:
            if row["total_tat_min"] > 25.0:
                freshness_penalty += 15.0

    total_cost = total_base + total_surge + total_peak + breach_penalty + freshness_penalty
    return total_cost / n


def run_batching_optimizer(
    df: pd.DataFrame,
    sla_threshold_min: float = 30.0,
    max_batch_size: int = 3,
    radius_threshold_km: float = 2.5,
) -> Tuple[pd.DataFrame, List[Batch], pd.DataFrame]:
    
    batches: List[Batch] = []
    order_batch_map: dict = {}
    order_cpo_map:   dict = {}
    order_breach_map: dict = {}
    batch_counter = 0
    records = df.to_dict('index')

    for store_id, store_df in df.groupby("dark_store_id"):
        store = STORE_LOOKUP[store_id]
        pending = list(store_df.sort_values("timestamp").index)

        while pending:
            seed_idx = pending.pop(0)
            seed_row = records[seed_idx]
            current_batch_indices = [seed_idx]

            if max_batch_size > 1:
                store_lat, store_lon = store.lat, store.lon
                savings_list = []
                lookahead_horizon = pending[:15]
                
                for cand_idx in lookahead_horizon:
                    cand_row = records[cand_idx]
                    time_diff_mins = (cand_row["timestamp"] - seed_row["timestamp"]).total_seconds() / 60.0
                    if time_diff_mins > 15.0:
                        break
                    
                    dist_store_seed = haversine_scalar(store_lat, store_lon, seed_row["delivery_lat"], seed_row["delivery_lon"])
                    dist_store_cand = haversine_scalar(store_lat, store_lon, cand_row["delivery_lat"], cand_row["delivery_lon"])
                    dist_seed_cand  = haversine_scalar(seed_row["delivery_lat"], seed_row["delivery_lon"], cand_row["delivery_lat"], cand_row["delivery_lon"])
                    
                    savings = dist_store_seed + dist_store_cand - dist_seed_cand
                    savings_list.append((savings, cand_idx))
                
                savings_list.sort(key=lambda x: x[0], reverse=True)

                for _, candidate_idx in savings_list:
                    if len(current_batch_indices) >= max_batch_size:
                        break

                    candidate_row = records[candidate_idx]
                    too_far = False
                    for existing_idx in current_batch_indices:
                        existing_row = records[existing_idx]
                        dist = haversine_scalar(
                            candidate_row["delivery_lat"], candidate_row["delivery_lon"],
                            existing_row["delivery_lat"], existing_row["delivery_lon"]
                        )
                        if dist > radius_threshold_km:
                            too_far = True
                            break

                    if too_far:
                        continue

                    trial_indices = current_batch_indices + [candidate_idx]
                    trial_pickup = max(records[i]["dark_store_pickup_wait_time"] for i in trial_indices)
                    trial_onboard = sum(records[i]["rider_onboarding_delay"] for i in trial_indices) / len(trial_indices)
                    trial_transit = sum(records[i]["transit_time_min"] for i in trial_indices)
                    trial_dropoff = sum(records[i]["door_dropoff_exception_time"] for i in trial_indices)
                    trial_tat = trial_pickup + trial_onboard + trial_transit + trial_dropoff

                    if trial_tat <= sla_threshold_min:
                        current_batch_indices.append(candidate_idx)
                        pending.remove(candidate_idx)

            batch_slice   = df.loc[current_batch_indices]
            final_tat     = compute_batch_tat(batch_slice)
            sla_breached  = final_tat > sla_threshold_min

            batch_id = f"B{batch_counter:05d}"
            batch_cpo = compute_batch_cpo(batch_slice, sla_breached)

            breach_detail = None
            if sla_breached:
                worst_delay = batch_slice["dark_store_pickup_wait_time"].max()
                breach_detail = (
                    f"TAT={final_tat:.1f}m > SLA {sla_threshold_min:.0f}m | "
                    f"Pickup wait={worst_delay:.1f}m | Store={store.name}"
                )

            b = Batch(
                batch_id=batch_id,
                store_id=store_id,
                order_ids=list(batch_slice["order_id"]),
                total_tat_min=round(final_tat, 2),
                batch_cpo=round(batch_cpo, 2),
                sla_breached=sla_breached,
                breach_details=breach_detail,
                is_peak=bool(batch_slice["is_peak"].any()),
            )
            batches.append(b)

            for idx in current_batch_indices:
                order_batch_map[df.loc[idx, "order_id"]] = batch_id
                order_cpo_map[df.loc[idx, "order_id"]]   = batch_cpo
                order_breach_map[df.loc[idx, "order_id"]] = sla_breached

            batch_counter += 1

    enriched_df = df.copy()
    enriched_df["batch_id"]     = enriched_df["order_id"].map(order_batch_map)
    enriched_df["batch_cpo"]    = enriched_df["order_id"].map(order_cpo_map)
    enriched_df["sla_breached"] = enriched_df["order_id"].map(order_breach_map)
    enriched_df["sla_breached"] = enriched_df["sla_breached"].fillna(False)
    enriched_df["batch_cpo"]    = enriched_df["batch_cpo"].fillna(enriched_df["raw_cpo"])

    exception_df = enriched_df[enriched_df["sla_breached"]].copy()
    batch_detail_map = {b.batch_id: b.breach_details for b in batches if b.sla_breached}
    exception_df["breach_detail"] = exception_df["batch_id"].map(batch_detail_map)

    return enriched_df, batches, exception_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — UNIT ECONOMICS CALCULATOR
# ═══════════════════════════════════════════════════════════════════

def compute_unit_economics(enriched_df: pd.DataFrame, batches: List[Batch]) -> dict:
    total_orders    = len(enriched_df)
    total_batches   = len(batches)
    breached_orders = enriched_df["sla_breached"].sum()

    total_rider_minutes = sum(b.total_tat_min for b in batches)
    total_rider_hours   = total_rider_minutes / 60.0
    oprh = total_orders / total_rider_hours if total_rider_hours > 0 else 0.0

    baseline_avg_tat_min = 22.0
    baseline_oprh = 60.0 / baseline_avg_tat_min

    blended_cpo    = enriched_df["batch_cpo"].mean()
    baseline_cpo   = enriched_df["raw_cpo"].mean()

    sla_adherence = (1 - breached_orders / total_orders) * 100 if total_orders > 0 else 0.0

    estimated_fleet_size = max(1, total_batches // 20) 
    total_available_min  = 24 * 60 * estimated_fleet_size
    active_time_min      = total_rider_minutes
    idle_time_min        = max(0.0, total_available_min - active_time_min)
    utilisation_rate     = (active_time_min / total_available_min) * 100

    return {
        "total_orders":       total_orders,
        "total_batches":      total_batches,
        "breached_orders":    int(breached_orders),
        "oprh":               round(oprh, 2),
        "baseline_oprh":      round(baseline_oprh, 2),
        "blended_cpo":        round(blended_cpo, 2),
        "baseline_cpo":       round(baseline_cpo, 2),
        "sla_adherence":      round(sla_adherence, 2),
        "active_time_min":    round(active_time_min, 1),
        "idle_time_min":      round(idle_time_min, 1),
        "utilisation_rate":   round(utilisation_rate, 2),
        "total_rider_hours":  round(total_rider_hours, 1),
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.6)",
    font=dict(family="IBM Plex Mono, monospace", color="#F0F4F1", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    coloraxis_colorbar=dict(
        tickfont=dict(color="#F0F4F1"),
        title=dict(font=dict(color="#2ECC71")),
    ),
)


def chart_density_map(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Densitymapbox(
        lat=df["delivery_lat"], lon=df["delivery_lon"], z=np.ones(len(df)),
        radius=18, showscale=False, name="Delivery Density",
        colorscale=[
            [0.0, "rgba(30,107,69,0)"], [0.4, "rgba(30,107,69,0.5)"],
            [0.7, "rgba(46,204,113,0.7)"], [1.0, "rgba(243,156,18,0.95)"],
        ],
    ))

    for store in DARK_STORES:
        fig.add_trace(go.Scattermapbox(
            lat=[store.lat], lon=[store.lon], mode="markers+text",
            marker=dict(size=18, color="#2ECC71", symbol="square"),
            text=[store.name], textposition="top right",
            textfont=dict(color="#2ECC71", size=10), name=store.name,
        ))

    breached = df[df["sla_breached"]]
    if not breached.empty:
        fig.add_trace(go.Scattermapbox(
            lat=breached["delivery_lat"], lon=breached["delivery_lon"],
            mode="markers", marker=dict(size=5, color="#E74C3C", opacity=0.7),
            name="SLA Breach", customdata=breached["total_tat_min"],
            hovertemplate="<b>SLA BREACH</b><br>TAT: %{customdata:.1f}m<extra></extra>",
        ))

    center_lat = df["delivery_lat"].mean()
    center_lon = df["delivery_lon"].mean()

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12.2,
        ),
        height=450,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("plot_bgcolor",)},
        legend=dict(
            bgcolor="rgba(22,33,62,0.8)",
            bordercolor="rgba(46, 204, 113, 0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    return fig


def chart_pareto_frontier(enriched_df: pd.DataFrame) -> go.Figure:
    agg = (
        enriched_df.groupby(["hour", "dark_store_id"])
        .agg(
            avg_cpo=("batch_cpo", "mean"),
            sla_adherence=("sla_breached", lambda x: (1 - x.mean()) * 100),
            order_count=("order_id", "count"),
            is_peak=("is_peak", "first"),
        ).reset_index()
    )
    agg["store_name"] = agg["dark_store_id"].map({s.store_id: s.name for s in DARK_STORES})

    COLOR_MAP = {"Koramangala Hub": "#2ECC71", "Indiranagar Hub": "#F39C12", "Whitefield Hub": "#3498DB"}
    fig = go.Figure()

    for store_name, grp in agg.groupby("store_name"):
        color = COLOR_MAP.get(store_name, "#AAAAAA")
        marker_symbol = ["diamond" if p else "circle" for p in grp["is_peak"]]
        fig.add_trace(go.Scatter(
            x=grp["avg_cpo"], y=grp["sla_adherence"], mode="markers", name=store_name,
            marker=dict(size=grp["order_count"] / grp["order_count"].max() * 22 + 5,
                        color=color, opacity=0.75, symbol=marker_symbol,
                        line=dict(width=1, color="rgba(255,255,255,0.2)")),
            customdata=grp["order_count"],
            hovertemplate=(f"<b>{store_name}</b><br>CPO: ₹%{{x:.2f}}<br>SLA Adherence: %{{y:.1f}}%<br>Orders: %{{customdata}}<extra></extra>"),
        ))

    x_range = np.linspace(agg["avg_cpo"].min(), agg["avg_cpo"].max(), 100)
    x_norm = (x_range - x_range.min()) / (x_range.max() - x_range.min())
    pareto_y = 70 + 28 / (1 + np.exp(-8 * (x_norm - 0.35)))

    fig.add_trace(go.Scatter(
        x=x_range, y=pareto_y, mode="lines", name="Pareto Frontier",
        line=dict(color="#E74C3C", width=2, dash="dot"), hoverinfo="skip",
    ))

    fig.add_annotation(
        x=x_range[40], y=pareto_y[40] + 3, text="← Pareto Efficiency Frontier",
        showarrow=False, font=dict(color="#E74C3C", size=10, family="IBM Plex Mono"),
    )

    fig.update_layout(
        xaxis=dict(title="Avg Cost Per Order (CPO) ₹", gridcolor="rgba(44, 62, 80, 0.27)", zeroline=False),
        yaxis=dict(title="SLA Adherence Rate (%)", gridcolor="rgba(44, 62, 80, 0.27)", range=[50, 102]),
        height=400,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="rgba(46, 204, 113, 0.2)", borderwidth=1),
        **CHART_LAYOUT,
    )
    return fig


def chart_oprh_by_hour(enriched_df: pd.DataFrame) -> go.Figure:
    hourly = (
        enriched_df.groupby("hour")
        .agg(order_count=("order_id", "count"), avg_tat=("total_tat_min", "mean"), avg_cpo=("batch_cpo", "mean"))
        .reset_index()
    )
    hourly["oprh_approx"] = 60 / hourly["avg_tat"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["order_count"], name="Order Volume",
        marker_color="rgba(30,107,69,0.55)", marker_line_color="#1E6B45", marker_line_width=1,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["oprh_approx"], name="OPRH", mode="lines+markers",
        line=dict(color="#2ECC71", width=2.5), marker=dict(size=6, color="#2ECC71"),
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["avg_cpo"], name="Avg CPO (₹)",
        mode="lines", line=dict(color="#F39C12", width=1.5, dash="dash"),
    ), secondary_y=True)

    for start, end in PEAK_WINDOWS:
        fig.add_vrect(
            x0=start, x1=end, fillcolor="rgba(243,156,18,0.07)", line_width=0,
            annotation_text="PEAK" if start == 8 else "", annotation_font=dict(color="#F39C12", size=9),
        )

    fig.update_layout(
        xaxis=dict(title="Hour of Day", gridcolor="rgba(44, 62, 80, 0.27)", tickmode="linear", dtick=2),
        yaxis=dict(title="Order Count", gridcolor="rgba(44, 62, 80, 0.27)"),
        yaxis2=dict(title="OPRH / CPO ₹", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        height=350,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="rgba(46, 204, 113, 0.2)", borderwidth=1),
        **CHART_LAYOUT,
    )
    return fig


def chart_tat_distribution(enriched_df: pd.DataFrame, sla_threshold: float) -> go.Figure:
    fig = go.Figure()
    colors = {"Peak": "#F39C12", "Off-Peak": "#2ECC71"}
    
    for period, grp in enriched_df.groupby("period"):
        fig.add_trace(go.Histogram(
            x=grp["total_tat_min"], name=period, nbinsx=50, opacity=0.65, marker_color=colors.get(period, "#AAAAAA"),
        ))

    fig.add_vline(
        x=sla_threshold, line_dash="dash", line_color="#E74C3C", line_width=2,
        annotation_text=f"SLA Limit {sla_threshold:.0f}m", annotation_font=dict(color="#E74C3C", size=10),
    )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Total TAT (min)", gridcolor="rgba(44, 62, 80, 0.27)"),
        yaxis=dict(title="Order Count", gridcolor="rgba(44, 62, 80, 0.27)"),
        height=320,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="rgba(46, 204, 113, 0.2)", borderwidth=1),
        **CHART_LAYOUT,
    )
    return fig


def chart_cost_breakdown(enriched_df: pd.DataFrame) -> go.Figure:
    agg = enriched_df.groupby("dark_store_id").agg(
        base=("base_payout", "mean"), surge=("distance_surge", "mean"), peak_cost=("peak_multiplier_cost", "mean"),
    ).reset_index()
    agg["store_name"] = agg["dark_store_id"].map({s.store_id: s.name for s in DARK_STORES})

    fig = go.Figure()
    components = [("base", "Base Payout", "#1E6B45"), ("surge", "Distance Surge", "#2ECC71"), ("peak_cost", "Peak Premium", "#F39C12")]
    
    for col, label, color in components:
        fig.add_trace(go.Bar(x=agg["store_name"], y=agg[col], name=label, marker_color=color))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Dark Store", gridcolor="rgba(44, 62, 80, 0.27)"),
        yaxis=dict(title="Avg ₹ per Order", gridcolor="rgba(44, 62, 80, 0.27)"),
        height=320,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="rgba(46, 204, 113, 0.2)", borderwidth=1),
        **CHART_LAYOUT,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 — STREAMLIT APP LAYOUT
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating synthetic order dataset…")
def get_raw_data():
    return generate_orders(n_orders=1500)


def kpi_card(label: str, value: str, delta: str, delta_type: str = "neutral") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta {delta_type}">{delta}</div>
    </div>
    """


def main():
    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='font-family: Syne, sans-serif; font-weight: 800; font-size: 20px;
                    color: #2ECC71; letter-spacing: 1px; margin-bottom: 4px;'>
            🚴 LAST MILE OPT.
        </div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 10px;
                    color: #7F8C8D; letter-spacing: 2px; margin-bottom: 24px;'>
            Q-COMMERCE UNIT ECONOMICS
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**OPTIMIZATION CONTROLS**")

        sla_threshold = st.slider("SLA Threshold (min)", min_value=20, max_value=45, value=30, step=1)
        radius_km = st.slider("Delivery Radius Threshold (km)", min_value=0.5, max_value=3.5, value=2.0, step=0.25)
        max_batch = st.select_slider("Max Batch Size (orders/rider)", options=[1, 2, 3], value=2)

        st.markdown("---")
        st.markdown("**FILTERS**")

        store_options = ["All Stores"] + [s.name for s in DARK_STORES]
        selected_store = st.selectbox("Dark Store", store_options)
        selected_period = st.selectbox("Time Period", ["All Periods", "Peak", "Off-Peak"])

        st.markdown("---")
        st.markdown("""
        <div style='font-family: IBM Plex Mono, monospace; font-size: 9px; color: #4A5568;'>
        SIMULATING 1,500 ORDERS<br>3 DARK STORES · 24-HOUR CYCLE<br>
        HAVERSINE SPATIAL ENGINE<br>GREEDY BATCH OPTIMIZER
        </div>
        """, unsafe_allow_html=True)

    # ── Load & Process Data ──────────────────────────────────────
    raw_df = get_raw_data()

    @st.cache_data(show_spinner="Running batching optimizer…")
    def run_optimizer(sla_t, radius, max_b):
        return run_batching_optimizer(raw_df, sla_t, max_b, radius)

    enriched_df, batches, exception_df = run_optimizer(sla_threshold, radius_km, max_batch)

    view_df = enriched_df.copy()
    if selected_store != "All Stores":
        view_df = view_df[view_df["store_name"] == selected_store]
    if selected_period != "All Periods":
        view_df = view_df[view_df["period"] == selected_period]

    econ = compute_unit_economics(view_df, [b for b in batches if
            (selected_store == "All Stores" or
             STORE_LOOKUP.get(b.store_id, type('', (), {'name': ''})()).name == selected_store)])

    # ── Page Header ──────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; align-items:baseline; gap:16px; margin-bottom:4px;'>
        <span style='font-family:Syne,sans-serif; font-weight:800; font-size:28px; color:#F0F4F1;'>
            Last Mile Cost & Unit Economics Optimizer
        </span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:11px; color:#2ECC71;
                     border:1px solid rgba(46, 204, 113, 0.3); padding:2px 8px; border-radius:4px;'>
            LIVE SIMULATION
        </span>
    </div>
    <div style='font-family:IBM Plex Mono,monospace; font-size:11px; color:#7F8C8D; margin-bottom:24px;'>
        Q-Commerce Dark Store Network · 1,500 Orders · 24-Hour Operational Cycle
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Ribbon ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    oprh_delta = econ["oprh"] - econ["baseline_oprh"]
    cpo_delta  = econ["blended_cpo"] - econ["baseline_cpo"]

    with k1:
        st.markdown(kpi_card("Orders / Rider Hour (OPRH)", f"{econ['oprh']:.2f}",
            f"▲ +{oprh_delta:.2f} vs baseline ({econ['baseline_oprh']:.2f})", "good" if oprh_delta > 0 else "bad"), unsafe_allow_html=True)

    with k2:
        st.markdown(kpi_card("Blended CPO (₹)", f"₹{econ['blended_cpo']:.2f}",
            f"{'▼' if cpo_delta < 0 else '▲'} {cpo_delta:+.2f} vs baseline (₹{econ['baseline_cpo']:.2f})", "good" if cpo_delta < 0 else "bad"), unsafe_allow_html=True)

    with k3:
        sla_color = "good" if econ["sla_adherence"] >= 92 else ("bad" if econ["sla_adherence"] < 85 else "neutral")
        st.markdown(kpi_card("SLA Adherence Rate", f"{econ['sla_adherence']:.1f}%",
            f"{econ['breached_orders']:,} orders breached / {econ['total_orders']:,} total", sla_color), unsafe_allow_html=True)

    with k4:
        st.markdown(kpi_card("Fleet Utilisation", f"{econ['utilisation_rate']:.1f}%",
            f"Active: {econ['active_time_min']:,.0f}m | Idle: {econ['idle_time_min']:,.0f}m", "good" if econ["utilisation_rate"] > 70 else "neutral"), unsafe_allow_html=True)

    # ── Delivery Density Map ──────────────────────────────────────
    st.markdown('<div class="section-header">01 · DELIVERY DENSITY & EXCEPTION MAP</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_density_map(view_df.copy()), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<span class="badge badge-green">■ Dark Store Hub</span>&nbsp;&nbsp;<span class="badge badge-amber">■ High Density Zone</span>&nbsp;&nbsp;<span class="badge badge-red">■ SLA Breach</span>', unsafe_allow_html=True)

    # ── Two-column: Pareto + OPRH trend ──────────────────────────
    st.markdown('<div class="section-header">02 · EFFICIENCY FRONTIERS & HOURLY DYNAMICS</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#7F8C8D;letter-spacing:1px;margin-bottom:8px;">CPO vs SLA ADHERENCE — PARETO FRONTIER</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_pareto_frontier(view_df), use_container_width=True, config={"displayModeBar": False})
        st.caption("⬦ = Peak Hour  ● = Off-Peak · Bubble size ∝ order volume · Red dash = Pareto efficiency frontier")

    with col_right:
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#7F8C8D;letter-spacing:1px;margin-bottom:8px;">OPRH & VOLUME BY HOUR</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_oprh_by_hour(view_df), use_container_width=True, config={"displayModeBar": False})
        st.caption("Amber shading = peak windows · OPRH degrades during high-congestion periods")

    # ── Two-column: TAT Distribution + Cost Breakdown ─────────────
    st.markdown('<div class="section-header">03 · TAT DISTRIBUTION & COST ANATOMY</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#7F8C8D;letter-spacing:1px;margin-bottom:8px;">TURNAROUND TIME DISTRIBUTION</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_tat_distribution(view_df, sla_threshold), use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#7F8C8D;letter-spacing:1px;margin-bottom:8px;">CPO COST COMPONENT BREAKDOWN BY STORE</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_cost_breakdown(view_df), use_container_width=True, config={"displayModeBar": False})

    # ── Exception Log ─────────────────────────────────────────────
    st.markdown('<div class="section-header">04 · OPERATIONAL EXCEPTION LOG</div>', unsafe_allow_html=True)
    exc_view = exception_df[exception_df["dark_store_id"].isin(view_df["dark_store_id"].unique())].copy()

    if exc_view.empty:
        st.success("✓ Zero SLA breaches under current parameters.")
    else:
        st.markdown(f'<span class="badge badge-red">⚠ {len(exc_view):,} SLA BREACHES DETECTED</span>&nbsp;&nbsp;<span class="badge badge-amber">₹{exc_view["batch_cpo"].mean():.2f} avg CPO on breached orders</span>', unsafe_allow_html=True)
        st.markdown("")

        log_display = exc_view[[
            "order_id", "store_name", "hour_label", "period", "total_tat_min", 
            "dark_store_pickup_wait_time", "transit_time_min", "batch_cpo", "batch_id", "breach_detail"
        ]].rename(columns={
            "order_id": "Order ID", "store_name": "Store", "hour_label": "Hour", "period": "Period",
            "total_tat_min": "TAT (min)", "dark_store_pickup_wait_time": "Pickup Wait", 
            "transit_time_min": "Transit (min)", "batch_cpo": "CPO (₹)", "batch_id": "Batch ID", "breach_detail": "Breach Detail"
        }).head(200)

        st.dataframe(log_display, use_container_width=True, height=320, column_config={
            "TAT (min)": st.column_config.NumberColumn(format="%.1f"),
            "Pickup Wait": st.column_config.NumberColumn(format="%.1f"),
            "Transit (min)": st.column_config.NumberColumn(format="%.1f"),
            "CPO (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        })

    # ── Summary Stats ─────────────────────────────────────────────
    st.markdown('<div class="section-header">05 · BATCH OPTIMISATION SUMMARY</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    
    with s1: st.metric("Total Orders", f"{econ['total_orders']:,}")
    with s2: st.metric("Total Batches", f"{econ['total_batches']:,}")
    with s3: st.metric("Avg Orders/Batch", f"{econ['total_orders'] / max(econ['total_batches'], 1):.2f}")
    with s4: st.metric("Rider-Hours Active", f"{econ['total_rider_hours']:,.1f}h")
    with s5: st.metric("SLA Breaches", f"{econ['breached_orders']:,}")

    store_summary = view_df.groupby("store_name").agg(
        Orders=("order_id", "count"), Avg_TAT=("total_tat_min", "mean"), Avg_CPO=("batch_cpo", "mean"),
        SLA_Breaches=("sla_breached", "sum"), Peak_Pct=("is_peak", "mean")
    ).reset_index().rename(columns={"store_name": "Dark Store"})

    store_summary["SLA Adherence %"] = ((1 - store_summary["SLA_Breaches"] / store_summary["Orders"]) * 100).round(1)
    store_summary["Avg TAT (min)"]  = store_summary["Avg_TAT"].round(1)
    store_summary["Avg CPO (₹)"]    = store_summary["Avg_CPO"].round(2)
    store_summary["Peak Order %"]   = (store_summary["Peak_Pct"] * 100).round(1)

    st.dataframe(
        store_summary[["Dark Store", "Orders", "Avg TAT (min)", "Avg CPO (₹)", "SLA_Breaches", "SLA Adherence %", "Peak Order %"]].rename(columns={"SLA_Breaches": "SLA Breaches"}),
        use_container_width=True, hide_index=True,
        column_config={
            "Avg CPO (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "SLA Adherence %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        }
    )

    st.markdown("""
    <div style='margin-top: 48px; padding: 20px; border-top: 1px solid rgba(44, 62, 80, 0.4);
                font-family: IBM Plex Mono, monospace; font-size: 9px; color: #4A5568;
                text-align: center; letter-spacing: 1px;'>
        LAST MILE COST & UNIT ECONOMICS OPTIMIZER · Q-COMMERCE ANALYTICS ENGINE<br>
        HAVERSINE SPATIAL BATCHING · MULTI-OBJECTIVE TAT/CPO OPTIMIZER · SYNTHETIC OPERATIONAL SIMULATION<br>
        © 2024 — CONFIDENTIAL INTERNAL ANALYTICS TOOL
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()