import os
import logging
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from dotenv import load_dotenv
from census import Census
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EVAdditionalFeaturesPipeline:

    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
    LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"
    SMART_URL = "https://edg.epa.gov/data/public/OA/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"

    # ✅ Stable FIPS for all 50 states + DC
    STATE_FIPS = {
        "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
        "DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19",
        "KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27",
        "MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35",
        "NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44",
        "SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53",
        "WV":"54","WI":"55","WY":"56"
    }

    ACS_VARS = {
        "population": "B01003_001E",
        "median_home_value": "B25077_001E",
        "median_income": "B19013_001E",
        "hh_100k_150k": "B19001_016E",
        "hh_150k_plus": "B19001_017E",
        "total_households": "B19001_001E",
        "bachelor": "B15003_022E",
        "masters_plus": "B15003_023E",
        "edu_total": "B15003_001E",
        "owner_occupied": "B25003_002E",
        "occupied_total": "B25003_001E",
        "single_detached": "B25024_002E",
        "housing_total": "B25024_001E",
        "drive_alone": "B08301_003E",
        "work_from_home": "B08301_021E",
        "commute_total": "B08301_001E"
    }

    def __init__(self):

        load_dotenv()
        census_key = os.getenv("CENSUS_API_KEY")

        if not census_key:
            raise ValueError("CENSUS_API_KEY not found in .env")

        self.census = Census(census_key)

        self.base_dir = "prediction_data"
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.final_dir = os.path.join(self.base_dir, "final")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

    # =====================================================
    # SAFE DOWNLOAD WITH RETRY
    # =====================================================
    def safe_download(self, url, output_path):

        if os.path.exists(output_path):
            logging.info(f"Already exists: {output_path}")
            return

        logging.info(f"Downloading: {url}")

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=3)
        session.mount("https://", HTTPAdapter(max_retries=retries))

        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info("Download complete.")

    # =====================================================
    # GET FIPS
    # =====================================================
    def get_state_fips(self, state_abbr):

        state_abbr = state_abbr.upper()

        if state_abbr not in self.STATE_FIPS:
            raise ValueError(f"Invalid state abbreviation: {state_abbr}")

        return self.STATE_FIPS[state_abbr]

    # =====================================================
    # TRACTS
    # =====================================================
    def get_tracts(self, state_abbr):

        state_fips = self.get_state_fips(state_abbr)

        folder = os.path.join(self.raw_dir, "tiger_tracts", state_abbr)
        os.makedirs(folder, exist_ok=True)

        zip_path = os.path.join(folder, f"{state_abbr}_tract.zip")
        shp_path = os.path.join(folder, f"tl_2020_{state_fips}_tract.shp")

        if not os.path.exists(shp_path):
            url = f"{self.TIGER_URL}tl_2020_{state_fips}_tract.zip"
            self.safe_download(url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(folder)

        gdf = gpd.read_file(shp_path)
        gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)

        return gdf.to_crs(4326)

    # =====================================================
    # ACS EXTENDED
    # =====================================================
    def get_acs(self, state_abbr):

        state_fips = self.get_state_fips(state_abbr)

        data = self.census.acs5.state_county_tract(
            list(self.ACS_VARS.values()),
            state_fips,
            Census.ALL,
            Census.ALL
        )

        df = pd.DataFrame(data)
        df["GEOID"] = (df["state"] + df["county"] + df["tract"]).str.zfill(11)
        df.rename(columns={v: k for k, v in self.ACS_VARS.items()}, inplace=True)

        # Percent Features
        df["pct_high_income"] = (
            df["hh_100k_150k"] + df["hh_150k_plus"]
        ) / (df["total_households"] + 1e-6)

        df["pct_bachelor_plus"] = (
            df["bachelor"] + df["masters_plus"]
        ) / (df["edu_total"] + 1e-6)

        df["pct_owner_occupied"] = (
            df["owner_occupied"] / (df["occupied_total"] + 1e-6)
        )

        df["pct_single_family"] = (
            df["single_detached"] / (df["housing_total"] + 1e-6)
        )

        df["pct_drive_alone"] = (
            df["drive_alone"] / (df["commute_total"] + 1e-6)
        )

        df["pct_work_from_home"] = (
            df["work_from_home"] / (df["commute_total"] + 1e-6)
        )

        return df

    # =====================================================
    # LODES
    # =====================================================
    def get_lodes(self, state_abbr):

        folder = os.path.join(self.raw_dir, "lodes")
        os.makedirs(folder, exist_ok=True)

        state_lower = state_abbr.lower()
        file_path = os.path.join(folder, f"{state_abbr}_wac.csv.gz")

        if not os.path.exists(file_path):
            url = f"{self.LODES_BASE}/{state_lower}/wac/{state_lower}_wac_S000_JT00_2021.csv.gz"
            self.safe_download(url, file_path)

        lodes = pd.read_csv(file_path, compression="gzip")

        lodes["GEOID"] = lodes["w_geocode"].astype(str).str.zfill(15).str[:11]

        lodes = lodes.groupby("GEOID").agg({
            "C000": "sum",
            "CE03": "sum"
        }).reset_index()

        lodes.rename(columns={
            "C000": "jobs_total",
            "CE03": "high_wage_jobs"
        }, inplace=True)

        lodes["high_wage_share"] = (
            lodes["high_wage_jobs"] / (lodes["jobs_total"] + 1e-6)
        )

        return lodes

    # =====================================================
    # SMART LOCATION (NATIONAL)
    # =====================================================
    def get_smart_location(self):

        folder = os.path.join(self.raw_dir, "smart_location")
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, "SmartLocationDatabaseV3.csv")

        if not os.path.exists(file_path):
            self.safe_download(self.SMART_URL, file_path)

        smart = pd.read_csv(file_path, low_memory=False)

        smart["GEOID"] = (
            smart["STATEFP"].astype(str).str.zfill(2) +
            smart["COUNTYFP"].astype(str).str.zfill(3) +
            smart["TRACTCE"].astype(str).str.zfill(6)
        )

        smart.replace([-99999, -999999], np.nan, inplace=True)

        smart_tract = (
            smart.groupby("GEOID")[[
                "D1A", "D3B", "D4A",
                "D1C", "D2A_JPHH",
                "NatWalkInd"
            ]]
            .mean()
            .reset_index()
        )

        return smart_tract

    # =====================================================
    # RUN SINGLE STATE
    # =====================================================
    def run_state(self, state_abbr):

        logging.info(f"Processing {state_abbr}")

        tracts = self.get_tracts(state_abbr)
        acs = self.get_acs(state_abbr)
        lodes = self.get_lodes(state_abbr)
        smart = self.get_smart_location()

        tracts = tracts.merge(acs, on="GEOID", how="left")
        tracts = tracts.merge(lodes, on="GEOID", how="left")
        tracts = tracts.merge(smart, on="GEOID", how="left")

        tracts.replace([-99999, -999999], np.nan, inplace=True)
        tracts = tracts[tracts["population"] > 0]

        geo_path = os.path.join(self.final_dir, f"ev_additional_{state_abbr}.geojson")
        csv_path = os.path.join(self.final_dir, f"ev_additional_{state_abbr}.csv")

        tracts.to_file(geo_path, driver="GeoJSON")
        tracts.drop(columns="geometry").to_csv(csv_path, index=False)

        logging.info(f"{state_abbr} dataset saved.")
        return tracts

    # =====================================================
    # RUN MULTIPLE STATES
    # =====================================================
    def run_multiple(self, state_list):

        all_states = []

        for state in state_list:
            df = self.run_state(state)
            df["state"] = state
            all_states.append(df)

        combined = pd.concat(all_states)

        geo_path = os.path.join(self.final_dir, "ev_additional_multi.geojson")
        csv_path = os.path.join(self.final_dir, "ev_additional_multi.csv")

        combined.to_file(geo_path, driver="GeoJSON")
        combined.drop(columns="geometry").to_csv(csv_path, index=False)

        logging.info("Multi-state dataset saved.")