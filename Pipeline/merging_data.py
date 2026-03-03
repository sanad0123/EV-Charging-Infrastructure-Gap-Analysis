import os
import logging
import pandas as pd
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EVMasterMergePipeline:

    def __init__(self):
        self.base_dir = "prediction_data"
        self.final_dir = os.path.join(self.base_dir, "final")

        if not os.path.exists(self.final_dir):
            raise ValueError("prediction_data/final directory not found.")

    # =====================================================
    # MERGE SINGLE STATE
    # =====================================================
    def merge_state(self, state_abbr):

        logging.info(f"Merging datasets for {state_abbr}")

        main_geo_path = os.path.join(
            self.final_dir,
            f"ev_dataset_{state_abbr}.geojson"
        )

        additional_csv_path = os.path.join(
            self.final_dir,
            f"ev_additional_{state_abbr}.csv"
        )

        main_gdf = gpd.read_file(main_geo_path)
        additional_df = pd.read_csv(additional_csv_path)

        # Force GEOID as string
        main_gdf["GEOID"] = main_gdf["GEOID"].astype(str).str.zfill(11)
        additional_df["GEOID"] = additional_df["GEOID"].astype(str).str.zfill(11)

        if "geometry" in additional_df.columns:
            additional_df = additional_df.drop(columns=["geometry"])

        merged = main_gdf.merge(
            additional_df,
            on="GEOID",
            how="left",
            suffixes=("", "_add")
        )

        # 🔥 Replace NaN with 0 (only numeric columns)
        numeric_cols = merged.select_dtypes(include=["float64", "int64"]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0)

        merged = merged.loc[:, ~merged.columns.duplicated()]

        geo_out = os.path.join(
            self.final_dir,
            f"ev_master_{state_abbr}.geojson"
        )

        csv_out = os.path.join(
            self.final_dir,
            f"ev_master_{state_abbr}.csv"
        )

        merged.to_file(geo_out, driver="GeoJSON")
        merged.drop(columns="geometry").to_csv(csv_out, index=False)

        logging.info(f"Master dataset saved for {state_abbr}")

        return merged
    # =====================================================
    # MERGE MULTIPLE STATES
    # =====================================================
    def merge_multiple(self, state_list):

        all_states = []

        for state in state_list:
            df = self.merge_state(state)
            df["state"] = state
            all_states.append(df)

        combined = gpd.GeoDataFrame(
            pd.concat(all_states, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326"
        )

        geo_out = os.path.join(self.final_dir, "ev_master_multi.geojson")
        csv_out = os.path.join(self.final_dir, "ev_master_multi.csv")

        combined.to_file(geo_out, driver="GeoJSON")
        combined.drop(columns="geometry").to_csv(csv_out, index=False)

        logging.info("Multi-state master dataset saved.")

        return combined