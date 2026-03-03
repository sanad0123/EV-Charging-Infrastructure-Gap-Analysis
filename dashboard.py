import os
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("EV Charging Infrastructure Gap Dashboard")

PREDICTED_DIR = "prediction_data/predicted"

# ------------------------------------------------
# LOAD STATES
# ------------------------------------------------
files = [
    f for f in os.listdir(PREDICTED_DIR)
    if f.startswith("ev_master_") and f.endswith("_with_predictions.geojson")
]

if not files:
    st.error("No predicted GeoJSON files found.")
    st.stop()

states = sorted([
    f.replace("ev_master_", "").replace("_with_predictions.geojson", "")
    for f in files
])

selected_state = st.selectbox("Select State", states)

file_path = os.path.join(
    PREDICTED_DIR,
    f"ev_master_{selected_state}_with_predictions.geojson"
)

gdf = gpd.read_file(file_path)

gdf["total_ports"] = gdf["total_ports"].fillna(0)
gdf["predicted_total_ports"] = gdf["predicted_total_ports"].fillna(0)
gdf["charging_gap"] = gdf["charging_gap"].fillna(0)

# ------------------------------------------------
# SUMMARY METRICS
# ------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Existing Ports", int(gdf["total_ports"].sum()))
col2.metric("Total Predicted Ports", int(gdf["predicted_total_ports"].sum()))
col3.metric("Overall Charging Gap", int(gdf["charging_gap"].sum()))

st.divider()

# ------------------------------------------------
# MAP
# ------------------------------------------------
st.subheader(f"Charging Status by Tract - {selected_state}")

center = [
    gdf.geometry.centroid.y.mean(),
    gdf.geometry.centroid.x.mean()
]

m = folium.Map(location=center, zoom_start=7)

# Define status
def get_status_color(gap):
    if gap > 0:
        return "#E74C3C"  # Red → Under-served
    elif gap < 0:
        return "#3498DB"  # Blue → Over-served
    else:
        return "#2ECC71"  # Green → Balanced

for _, row in gdf.iterrows():
    folium.GeoJson(
        row["geometry"],
        style_function=lambda feature, gap=row["charging_gap"]: {
            "fillColor": get_status_color(gap),
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.6,
        },
        tooltip=folium.Tooltip(
            f"""
            GEOID: {row['GEOID']}<br>
            Total Ports: {int(row['total_ports'])}<br>
            Predicted: {int(row['predicted_total_ports'])}<br>
            Gap: {int(row['charging_gap'])}
            """
        )
    ).add_to(m)

# ------------------------------------------------
# LEGEND
# ------------------------------------------------
legend_html = """
<div style="
position: fixed;
bottom: 50px; left: 50px;
width: 220px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding: 10px;
">
<b>Charging Status</b><br><br>
<span style="color:#E74C3C;">&#9632;</span> Under-Served (Gap > 0)<br>
<span style="color:#2ECC71;">&#9632;</span> Balanced (Gap = 0)<br>
<span style="color:#3498DB;">&#9632;</span> Over-Served (Gap < 0)
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=1200, height=600)

st.divider()

# ------------------------------------------------
# RANKINGS
# ------------------------------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.subheader("🔴 Top Under-Served")
    underserved = (
        gdf[gdf["charging_gap"] > 0]
        .sort_values("charging_gap", ascending=False)
        [["GEOID", "total_ports",
          "predicted_total_ports", "charging_gap"]]
        .head(15)
    )
    st.dataframe(underserved, use_container_width=True)

with colB:
    st.subheader("🟢 Balanced")
    balanced = (
        gdf[gdf["charging_gap"] == 0]
        [["GEOID", "total_ports",
          "predicted_total_ports", "charging_gap"]]
        .head(15)
    )
    st.dataframe(balanced, use_container_width=True)

with colC:
    st.subheader("🔵 Top Over-Served")
    overserved = (
        gdf[gdf["charging_gap"] < 0]
        .sort_values("charging_gap")
        [["GEOID", "total_ports",
          "predicted_total_ports", "charging_gap"]]
        .head(15)
    )
    st.dataframe(overserved, use_container_width=True)