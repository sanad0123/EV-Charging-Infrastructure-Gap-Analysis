import os
import numpy as np
import geopandas as gpd
import joblib

# Import your existing pipelines
from main_data import EVFullPipeline
from additional_data import EVAdditionalFeaturesPipeline
from merging_data import EVMasterMergePipeline


class EVCompleteStatePipeline:

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(self, model_path="models/xgb_total_ports_reduced.pkl"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)

        self.main_pipeline = EVFullPipeline()
        self.additional_pipeline = EVAdditionalFeaturesPipeline()
        self.merge_pipeline = EVMasterMergePipeline()

        # Stable FIPS mapping
        self.STATE_FIPS = {
            "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
            "DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19",
            "KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27",
            "MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35",
            "NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44",
            "SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53",
            "WV":"54","WI":"55","WY":"56"
        }

    # ==========================================================
    # SAFE LOG
    # ==========================================================
    def safe_log(self, x):
        return np.log1p(x.clip(lower=0))

    # ==========================================================
    # FEATURE ENGINEERING (MATCH TRAINING)
    # ==========================================================
    def create_features(self, df):

        df = df.copy()

        # Log transforms
        df["log_population"] = self.safe_log(df["population"])
        df["log_vehicles_total"] = self.safe_log(df["vehicles_total"])
        df["log_establishments_total"] = self.safe_log(df["establishments_total"])
        df["log_retail_establishments"] = self.safe_log(df["retail_establishments"])
        df["log_food_establishments"] = self.safe_log(df["food_establishments"])
        df["log_service_establishments"] = self.safe_log(df["service_establishments"])
        df["log_dist_nearest_charger_km"] = self.safe_log(df["dist_nearest_charger_km"])
        df["log_dist_nearest_dcfc_km"] = self.safe_log(df["dist_nearest_dcfc_km"])

        # Destination intensity (rank-based)
        df["destination_intensity"] = (
            np.log1p(df["jobs_total"]).rank(pct=True) +
            df["log_retail_establishments"].rank(pct=True) +
            df["log_food_establishments"].rank(pct=True)
        ) / 3

        # Urban intensity
        df["urban_intensity_index"] = (
            df["D1A"].rank(pct=True) +
            df["D3B"].rank(pct=True) +
            df["NatWalkInd"].rank(pct=True)
        ) / 3

        # Charging access gap
        df["charging_access_gap"] = (
            df["log_dist_nearest_charger_km"] +
            df["log_dist_nearest_dcfc_km"]
        )

        # Fill numeric NaNs
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    # ==========================================================
    # RUN FULL PIPELINE FOR STATE
    # ==========================================================
    def run(self, state):

        state = state.upper()

        if state not in self.STATE_FIPS:
            raise ValueError("Invalid state abbreviation")

        state_fips = self.STATE_FIPS[state]

        print(f"\n========== PROCESSING {state} ==========\n")

        # 1️⃣ Main pipeline
        print("Running main data pipeline...")
        self.main_pipeline.run_state(state, state_fips)

        # 2️⃣ Additional features
        print("Running additional features pipeline...")
        self.additional_pipeline.run_state(state)

        # 3️⃣ Merge
        print("Merging datasets...")
        merged_gdf = self.merge_pipeline.merge_state(state)

        # 4️⃣ Feature engineering
        print("Creating model features...")
        merged_gdf = self.create_features(merged_gdf)

        # 5️⃣ Prediction
        top_features = [
            "jobs_total",
            "destination_intensity",
            "log_dist_nearest_charger_km",
            "ALAND",
            "charging_access_gap",
            "log_dist_nearest_dcfc_km",
            "mean_commute_time",
            "D1A",
            "high_wage_share",
            "D1C",
            "log_food_establishments",
            "log_service_establishments",
            "D2A_JPHH",
            "pct_drive_alone",
            "pct_single_family"
        ]

        missing = [f for f in top_features if f not in merged_gdf.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = merged_gdf[top_features]

        print("Predicting total ports...")
        preds = self.model.predict(X)
        preds = np.clip(preds, 0, None)
        preds = np.ceil(preds).astype(int)

        merged_gdf["predicted_total_ports"] = preds
        merged_gdf["charging_gap"] = (
            merged_gdf["predicted_total_ports"] -
            merged_gdf["total_ports"]
        )

        # 6️⃣ Save GeoJSON
        output_dir = "prediction_data/predicted"
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/ev_master_{state}_with_predictions.geojson"
        merged_gdf.to_file(output_path, driver="GeoJSON")

        print("\nSaved:", output_path)

        # 7️⃣ Summary
        print("\n----- SUMMARY -----")
        print("Total Actual Ports:", merged_gdf["total_ports"].sum())
        print("Total Predicted Ports:", merged_gdf["predicted_total_ports"].sum())
        print("Overall Gap:", merged_gdf["charging_gap"].sum())

        return merged_gdf