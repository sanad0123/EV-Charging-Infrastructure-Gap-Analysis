import os
import io
import requests
import zipfile
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from dotenv import load_dotenv
from census import Census
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EVFullPipeline:

    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
    ZCTA_URL = "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip"
    ZBP_URL = "https://www2.census.gov/programs-surveys/cbp/datasets/2023/zbp23detail.zip"

    ACS_VARS = {
        "population": "B01003_001E",
        "households": "B11001_001E",
        "median_income": "B19013_001E",
        "workers": "B23025_002E",
        "vehicles_total": "B08201_001E",
        "mean_commute_time": "B08303_001E"  
    }

    def __init__(self):
        load_dotenv()

        self.census = Census(os.getenv("CENSUS_API_KEY"))
        self.nrel_key = os.getenv("NREL_API_KEY")

        self.base_dir = "prediction_data"
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.final_dir = os.path.join(self.base_dir, "final")

        for d in [self.raw_dir, self.processed_dir, self.final_dir]:
            os.makedirs(d, exist_ok=True)

    # =====================================================
    # 1️⃣ TRACTS
    # =====================================================
    def get_tracts(self, state_abbr, state_fips):
        folder = os.path.join(self.raw_dir, "tiger_tracts", state_abbr)
        os.makedirs(folder, exist_ok=True)

        shp_file = os.path.join(folder, f"tl_2020_{state_fips}_tract.shp")

        if not os.path.exists(shp_file):
            logging.info(f"Downloading TIGER tracts for {state_abbr}")
            url = f"{self.TIGER_URL}tl_2020_{state_fips}_tract.zip"
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(folder)

        gdf = gpd.read_file(shp_file)[["GEOID", "geometry"]]
        return gdf.to_crs(4326)

    # =====================================================
    # 2️⃣ ACS
    # =====================================================
    def get_acs(self, state_fips):
        data = self.census.acs5.state_county_tract(
            fields=list(self.ACS_VARS.values()),
            state_fips=state_fips,
            county_fips="*",
            tract="*"
        )

        df = pd.DataFrame(data)
        df["GEOID"] = df["state"] + df["county"] + df["tract"]

        df.rename(columns={v: k for k, v in self.ACS_VARS.items()}, inplace=True)

        for col in self.ACS_VARS.keys():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace(-666666666, np.nan, inplace=True)

        return df[["GEOID"] + list(self.ACS_VARS.keys())]

    # =====================================================
    # 3️⃣ NREL (NATIONAL REUSED)
    # =====================================================
    def get_nrel(self):
        path = os.path.join(self.raw_dir, "nrel", "nrel_ev.geojson")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            logging.info("Downloading NREL EV stations (national)")

            url = f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?fuel_type=ELEC&country=US&limit=all&api_key={self.nrel_key}"
            r = requests.get(url)
            stations = r.json()["fuel_stations"]

            df = pd.DataFrame(stations)
            df = df[df["access_code"] == "public"]

            df["geometry"] = df.apply(
                lambda x: Point(x.longitude, x.latitude), axis=1
            )

            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            gdf.to_file(path, driver="GeoJSON")

        return gpd.read_file(path)

    # =====================================================
    # 4️⃣ CHARGER SPATIAL AGGREGATION
    # =====================================================
    def aggregate_chargers(self, tracts, chargers):

    # Keep only needed columns
        chargers = chargers[[
            "geometry",
            "ev_level2_evse_num",
            "ev_dc_fast_num"
        ]]

        # Fill missing with 0
        chargers["ev_level2_evse_num"] = chargers["ev_level2_evse_num"].fillna(0)
        chargers["ev_dc_fast_num"] = chargers["ev_dc_fast_num"].fillna(0)

        # Spatial join
        joined = gpd.sjoin(chargers, tracts, predicate="within")

        # Aggregate per tract
        agg = joined.groupby("GEOID").agg({
            "ev_level2_evse_num": "sum",
            "ev_dc_fast_num": "sum"
        }).reset_index()

        agg.rename(columns={
            "ev_level2_evse_num": "l2_ports",
            "ev_dc_fast_num": "dcfc_ports"
        }, inplace=True)

        # Add total ports
        agg["total_ports"] = agg["l2_ports"] + agg["dcfc_ports"]

        # Station count
        station_count = joined.groupby("GEOID").size().reset_index(name="station_count")

        agg = agg.merge(station_count, on="GEOID", how="left")

        # Merge back to tracts
        tracts = tracts.merge(agg, on="GEOID", how="left")

        # Fill NaN
        tracts[["l2_ports", "dcfc_ports", "total_ports", "station_count"]] = \
            tracts[["l2_ports", "dcfc_ports", "total_ports", "station_count"]].fillna(0)

        return tracts

    # =====================================================
    # 5️⃣ DISTANCE CALCULATION
    # =====================================================
    def compute_distances(self, tracts, chargers):
    

        # ---------------------------
        # Project to meters
        # ---------------------------
        tracts_proj = tracts.to_crs(5070)
        chargers_proj = chargers.to_crs(5070)

        # ---------------------------
        # Create centroids
        # ---------------------------
        centroids = tracts_proj.geometry.centroid
        tract_coords = np.array(list(zip(centroids.x, centroids.y)))

        # ---------------------------
        # All chargers
        # ---------------------------
        charger_coords = np.array(
            list(zip(chargers_proj.geometry.x, chargers_proj.geometry.y))
        )

        tree_all = cKDTree(charger_coords)
        dist_all, _ = tree_all.query(tract_coords, k=1)

        tracts["dist_nearest_charger_km"] = dist_all / 1000

        # ---------------------------
        # DC Fast chargers only
        # ---------------------------
        chargers_dcfc = chargers_proj[
            chargers_proj["ev_dc_fast_num"] > 0
        ]

        if len(chargers_dcfc) > 0:

            dcfc_coords = np.array(
                list(zip(chargers_dcfc.geometry.x, chargers_dcfc.geometry.y))
            )

            tree_dcfc = cKDTree(dcfc_coords)
            dist_dcfc, _ = tree_dcfc.query(tract_coords, k=1)

            tracts["dist_nearest_dcfc_km"] = dist_dcfc / 1000

        else:
            tracts["dist_nearest_dcfc_km"] = np.nan

        return tracts

    # =====================================================
    # 6️⃣ ZCTA (REUSED NATIONAL)
    # =====================================================
    def get_zcta(self):
        folder = os.path.join(self.raw_dir, "zcta")
        shp_file = os.path.join(folder, "tl_2020_us_zcta520.shp")
        os.makedirs(folder, exist_ok=True)

        if not os.path.exists(shp_file):
            logging.info("Downloading national ZCTA shapefile")
            r = requests.get(self.ZCTA_URL)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(folder)

        gdf = gpd.read_file(shp_file)[["ZCTA5CE20", "geometry"]]
        gdf.rename(columns={"ZCTA5CE20": "ZIP"}, inplace=True)

        return gdf.to_crs(5070)

    # =====================================================
    # 7️⃣ ZBP (REUSED NATIONAL)
    # =====================================================
    def get_zbp(self):
        folder = os.path.join(self.raw_dir, "zbp")
        file_path = os.path.join(folder, "zbp23detail.txt")
        os.makedirs(folder, exist_ok=True)

        if not os.path.exists(file_path):
            logging.info("Downloading ZBP 2023 data")
            r = requests.get(self.ZBP_URL)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(folder)

        df = pd.read_csv(file_path)

        df = df[["zip", "naics", "est"]]
        df["zip"] = df["zip"].astype(str).str.zfill(5)

        df["category"] = "other"
        df.loc[df["naics"].str.startswith(("44", "45")), "category"] = "retail"
        df.loc[df["naics"].str.startswith("72"), "category"] = "food"
        df.loc[df["naics"].str.startswith(("54", "81")), "category"] = "service"

        pivot = df.pivot_table(
            index="zip",
            columns="category",
            values="est",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        pivot.rename(columns={
            "zip": "ZIP",
            "other": "establishments_total",
            "retail": "retail_establishments",
            "food": "food_establishments",
            "service": "service_establishments"
        }, inplace=True)

        return pivot

    # =====================================================
    # 8️⃣ ZBP AREA-WEIGHTED ALLOCATION
    # =====================================================
    def allocate_zbp_to_tracts(self, tracts):
        logging.info("Allocating ZBP to tracts")

        tracts_proj = tracts.to_crs(5070)
        zcta = self.get_zcta()
        zbp = self.get_zbp()

        zcta = zcta.merge(zbp, on="ZIP", how="left").fillna(0)
        zcta["zcta_area"] = zcta.geometry.area

        overlay = gpd.overlay(tracts_proj, zcta, how="intersection")
        overlay["intersection_area"] = overlay.geometry.area
        overlay["weight"] = overlay["intersection_area"] / overlay["zcta_area"]

        business_cols = [
            "establishments_total",
            "retail_establishments",
            "food_establishments",
            "service_establishments"
        ]

        for col in business_cols:
            overlay[col] = overlay[col] * overlay["weight"]

        agg = overlay.groupby("GEOID")[business_cols].sum().reset_index()

        tracts = tracts.merge(agg, on="GEOID", how="left").fillna(0)

        return tracts

    # =====================================================
    # 9️⃣ RUN SINGLE STATE
    # =====================================================
    def run_state(self, state_abbr, state_fips):

        logging.info(f"Processing {state_abbr}")

        tracts = self.get_tracts(state_abbr, state_fips)
        acs = self.get_acs(state_fips)
        chargers = self.get_nrel()

        tracts = tracts.merge(acs, on="GEOID", how="left")
        tracts = self.aggregate_chargers(tracts, chargers)
        tracts = self.compute_distances(tracts, chargers)
        tracts = self.allocate_zbp_to_tracts(tracts)

        geojson_path = os.path.join(self.final_dir, f"ev_dataset_{state_abbr}.geojson")
        csv_path = os.path.join(self.final_dir, f"ev_dataset_{state_abbr}.csv")

        tracts.to_file(geojson_path, driver="GeoJSON")
        tracts.drop(columns="geometry").to_csv(csv_path, index=False)

        logging.info(f"{state_abbr} dataset saved.")
        return tracts

    # =====================================================
    # 🔟 RUN MULTIPLE STATES
    # =====================================================
    def run_multiple(self, states_dict):

        all_states = []

        for abbr, fips in states_dict.items():
            df = self.run_state(abbr, fips)
            df["state"] = abbr
            all_states.append(df)

        combined = pd.concat(all_states)

        geo_path = os.path.join(self.final_dir, "ev_dataset_multi.geojson")
        csv_path = os.path.join(self.final_dir, "ev_dataset_multi.csv")

        combined.to_file(geo_path, driver="GeoJSON")
        combined.drop(columns="geometry").to_csv(csv_path, index=False)

        logging.info("Multi-state dataset saved.")