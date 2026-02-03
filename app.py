"""
Phase-II: Robotic Decision Support & Environment Optimization
Streamlit UI for Crop Yield Prediction and Robot Action Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from itertools import product

# --- Configuration ---
MODEL_DIR = "models/production"
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion_xgboost_pipeline.joblib")
ROLLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "rollback_extratrees_pipeline.joblib")

# Define options for categorical features (from actual training data)
CROP_OPTIONS = [
    "Arecanut",
    "Arhar/Tur",
    "Bajra",
    "Banana",
    "Barley",
    "Black pepper",
    "Cardamom",
    "Cashewnut",
    "Castor seed",
    "Coconut",
    "Coriander",
    "Cotton(lint)",
    "Cowpea(Lobia)",
    "Dry chillies",
    "Garlic",
    "Ginger",
    "Gram",
    "Groundnut",
    "Guar seed",
    "Horse-gram",
    "Jowar",
    "Jute",
    "Khesari",
    "Linseed",
    "Maize",
    "Masoor",
    "Mesta",
    "Moong(Green Gram)",
    "Moth",
    "Niger seed",
    "Oilseeds total",
    "Onion",
    "Other  Rabi pulses",
    "Other Cereals",
    "Other Kharif pulses",
    "Other Summer Pulses",
    "Peas & beans (Pulses)",
    "Potato",
    "Ragi",
    "Rapeseed &Mustard",
    "Rice",
    "Safflower",
    "Sannhamp",
    "Sesamum",
    "Small millets",
    "Soyabean",
    "Sugarcane",
    "Sunflower",
    "Sweet potato",
    "Tapioca",
    "Tobacco",
    "Turmeric",
    "Urad",
    "Wheat",
    "other oilseeds",
]
SEASON_OPTIONS = ["Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"]
STATE_OPTIONS = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

# Variation ranges for environment optimization
VARIATION_CONFIG = {
    "Fertilizer_per_Area": {
        "steps": [-20, -10, 0, 10, 20],  # kg/hectare variations
        "unit": "kg/ha",
        "action_name": "Fertilizer Application",
    },
    "Pesticide_per_Area": {
        "steps": [-0.1, -0.05, 0, 0.05, 0.1],  # variations
        "unit": "units/ha",
        "action_name": "Pesticide Application",
    },
    "Annual_Rainfall": {
        "steps": [-100, -50, 0, 50, 100],  # mm variations (simulated via irrigation)
        "unit": "mm (irrigation equivalent)",
        "action_name": "Irrigation",
    },
}


@st.cache_resource
def load_models():
    """Load production models with caching."""
    models = {}

    if os.path.exists(CHAMPION_MODEL_PATH):
        models["champion"] = joblib.load(CHAMPION_MODEL_PATH)
        models["champion_name"] = "XGBoost (Champion)"

    if os.path.exists(ROLLBACK_MODEL_PATH):
        models["rollback"] = joblib.load(ROLLBACK_MODEL_PATH)
        models["rollback_name"] = "Extra Trees (Rollback)"

    if not models:
        st.error("No production models found! Please train models first.")
        return None

    return models


def generate_environment_variations(base_input: dict) -> pd.DataFrame:
    """
    Generate all possible combinations of environmental variations.
    Creates permutations of adjustable parameters around the base values.
    """
    # Get variation steps for each adjustable parameter
    variation_lists = {}

    for param, config in VARIATION_CONFIG.items():
        base_value = base_input[param]
        # Generate absolute values from variations
        varied_values = [max(0, base_value + step) for step in config["steps"]]
        # Remove duplicates and sort
        varied_values = sorted(list(set(varied_values)))
        variation_lists[param] = varied_values

    # Generate all combinations using itertools.product (permutation of all possibilities)
    param_names = list(variation_lists.keys())
    param_values = [variation_lists[name] for name in param_names]

    combinations = list(product(*param_values))

    # Build DataFrame with all combinations
    records = []
    for combo in combinations:
        record = base_input.copy()
        for i, param_name in enumerate(param_names):
            record[param_name] = combo[i]
        records.append(record)

    return pd.DataFrame(records)


def predict_yields(model, variations_df: pd.DataFrame) -> np.ndarray:
    """Run predictions on all environmental variations."""
    return model.predict(variations_df)


def find_optimal_configuration(
    variations_df: pd.DataFrame, predictions: np.ndarray
) -> tuple:
    """Find the configuration with maximum predicted yield."""
    optimal_idx = np.argmax(predictions)
    optimal_config = variations_df.iloc[optimal_idx].to_dict()
    optimal_yield = predictions[optimal_idx]
    return optimal_config, optimal_yield, optimal_idx


def generate_robot_recommendations(base_input: dict, optimal_config: dict) -> list:
    """
    Generate actionable recommendations for the robot decision system.
    Compares optimal configuration with current conditions.
    """
    recommendations = []

    for param, config in VARIATION_CONFIG.items():
        base_val = base_input[param]
        optimal_val = optimal_config[param]
        diff = optimal_val - base_val

        if abs(diff) > 0.001:  # Threshold for meaningful difference
            action = {
                "parameter": param,
                "action_name": config["action_name"],
                "current_value": base_val,
                "recommended_value": optimal_val,
                "change": diff,
                "unit": config["unit"],
                "direction": "increase" if diff > 0 else "decrease",
            }
            recommendations.append(action)

    return recommendations


def display_recommendations(
    recommendations: list, base_yield: float, optimal_yield: float
):
    """Display robot action recommendations in a user-friendly format."""

    yield_improvement = optimal_yield - base_yield
    yield_improvement_pct = (
        (yield_improvement / base_yield * 100) if base_yield > 0 else 0
    )

    st.subheader("Robot Decision Support System")

    # Yield improvement summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Predicted Yield", f"{base_yield:.4f}")
    with col2:
        st.metric("Optimal Predicted Yield", f"{optimal_yield:.4f}")
    with col3:
        st.metric(
            "Potential Improvement",
            f"{yield_improvement_pct:.2f}%",
            delta=f"{yield_improvement:.4f}",
        )

    st.divider()

    if not recommendations:
        st.success("Current conditions are already optimal! No changes recommended.")
        return

    st.markdown("### Recommended Actions")

    for rec in recommendations:
        with st.container():
            icon = "⬆️" if rec["direction"] == "increase" else "⬇️"

            st.markdown(f"""
            **{icon} {rec["action_name"]}**
            - Current: `{rec["current_value"]:.2f}` {rec["unit"]}
            - Recommended: `{rec["recommended_value"]:.2f}` {rec["unit"]}
            - Action: **{rec["direction"].upper()}** by `{abs(rec["change"]):.2f}` {rec["unit"]}
            """)

            # Specific robot instructions
            if "Irrigation" in rec["action_name"]:
                if rec["direction"] == "increase":
                    st.info(
                        "**Irrigation Robot**: Increase water supply to field sections."
                    )
                else:
                    st.info("**Irrigation Robot**: Reduce watering frequency.")

            elif "Fertilizer" in rec["action_name"]:
                if rec["direction"] == "increase":
                    st.info(
                        "**Fertilizer Robot**: Apply additional fertilizer at recommended rate."
                    )
                else:
                    st.info(
                        "**Fertilizer Robot**: Reduce fertilizer application in next cycle."
                    )

            elif "Pesticide" in rec["action_name"]:
                if rec["direction"] == "increase":
                    st.info("**Pesticide Robot**: Increase pest control measures.")
                else:
                    st.info(
                        "**Pesticide Robot**: Reduce pesticide usage - current levels sufficient."
                    )

            st.divider()


def main():
    st.set_page_config(page_title="Crop Yield Optimizer", page_icon="🌾", layout="wide")

    st.title("Crop Yield Prediction & Robot Decision Support")
    st.markdown("""
    **Phase-II: Robotic Decision Support & Environment Optimization**
    
    This system helps optimize crop yield by:
    1. Taking current field conditions as input
    2. Generating environmental variations (fertilizer, irrigation, pesticide)
    3. Predicting yield for all possible configurations
    4. Recommending optimal actions for agricultural robots
    """)

    st.divider()

    # Load models
    models = load_models()
    if models is None:
        st.stop()

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        options=["champion", "rollback"] if "rollback" in models else ["champion"],
        format_func=lambda x: models.get(f"{x}_name", x),
    )
    active_model = models[model_choice]

    st.sidebar.success(f"Using: {models[f'{model_choice}_name']}")

    # --- Input Section ---
    st.header("Enter Current Field Conditions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crop Information")
        crop = st.selectbox("Crop Type", options=CROP_OPTIONS, index=0)
        season = st.selectbox("Season", options=SEASON_OPTIONS, index=0)
        state = st.selectbox("State", options=STATE_OPTIONS, index=0)
        area = st.number_input(
            "Area (hectares)",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
        )

    with col2:
        st.subheader("Environmental Parameters")
        annual_rainfall = st.number_input(
            "Annual Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=800.0,
            step=50.0,
            help="Current/expected annual rainfall in millimeters",
        )
        fertilizer = st.number_input(
            "Fertilizer per Area (kg/hectare)",
            min_value=0.0,
            max_value=500.0,
            value=120.0,
            step=5.0,
            help="Current fertilizer application rate",
        )
        pesticide = st.number_input(
            "Pesticide per Area (units/hectare)",
            min_value=0.0,
            max_value=10.0,
            value=0.3,
            step=0.05,
            help="Current pesticide application rate",
        )

    # Build base input
    base_input = {
        "Crop": crop,
        "Season": season,
        "State": state,
        "Area": area,
        "Annual_Rainfall": annual_rainfall,
        "Fertilizer_per_Area": fertilizer,
        "Pesticide_per_Area": pesticide,
    }

    st.divider()

    # --- Optimization Settings ---
    with st.expander("Optimization Settings", expanded=False):
        st.markdown("Adjust variation ranges for environment optimization:")

        col1, col2, col3 = st.columns(3)

        with col1:
            fert_range = st.slider(
                "Fertilizer variation (±kg/ha)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )
            VARIATION_CONFIG["Fertilizer_per_Area"]["steps"] = [
                -fert_range,
                -fert_range // 2,
                0,
                fert_range // 2,
                fert_range,
            ]

        with col2:
            pest_range = st.slider(
                "Pesticide variation (±units/ha)",
                min_value=0.02,
                max_value=0.2,
                value=0.1,
                step=0.02,
            )
            VARIATION_CONFIG["Pesticide_per_Area"]["steps"] = [
                -pest_range,
                -pest_range / 2,
                0,
                pest_range / 2,
                pest_range,
            ]

        with col3:
            rain_range = st.slider(
                "Irrigation variation (±mm)",
                min_value=25,
                max_value=200,
                value=100,
                step=25,
            )
            VARIATION_CONFIG["Annual_Rainfall"]["steps"] = [
                -rain_range,
                -rain_range // 2,
                0,
                rain_range // 2,
                rain_range,
            ]

    # --- Run Optimization ---
    if st.button(
        "Optimize & Generate Recommendations",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Generating environmental variations..."):
            variations_df = generate_environment_variations(base_input)
            st.info(f"Generated {len(variations_df)} environmental configurations")

        with st.spinner("Running predictions on all configurations..."):
            predictions = predict_yields(active_model, variations_df)

        with st.spinner("Finding optimal configuration..."):
            optimal_config, optimal_yield, optimal_idx = find_optimal_configuration(
                variations_df, predictions
            )

        # Get base prediction
        base_df = pd.DataFrame([base_input])
        base_yield = active_model.predict(base_df)[0]

        # Generate recommendations
        recommendations = generate_robot_recommendations(base_input, optimal_config)

        st.divider()

        # Display results
        display_recommendations(recommendations, base_yield, optimal_yield)

        # Show detailed results in expandable section
        with st.expander("Detailed Analysis", expanded=False):
            st.subheader("All Configurations & Predictions")

            results_df = variations_df.copy()
            results_df["Predicted_Yield"] = predictions
            results_df = results_df.sort_values("Predicted_Yield", ascending=False)

            # Highlight optimal row
            st.dataframe(
                results_df.head(20).style.highlight_max(
                    subset=["Predicted_Yield"], color="lightgreen"
                ),
                use_container_width=True,
            )

            st.subheader("Optimal Configuration Details")
            opt_df = pd.DataFrame([optimal_config])
            opt_df["Predicted_Yield"] = optimal_yield
            st.dataframe(opt_df, use_container_width=True)

            # Visualization
            st.subheader("Yield Distribution Across Configurations")

            import plotly.express as px

            fig = px.histogram(
                results_df,
                x="Predicted_Yield",
                nbins=30,
                title="Distribution of Predicted Yields",
                labels={"Predicted_Yield": "Predicted Yield"},
            )
            fig.add_vline(
                x=base_yield,
                line_dash="dash",
                line_color="red",
                annotation_text="Current",
            )
            fig.add_vline(
                x=optimal_yield,
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal",
            )
            st.plotly_chart(fig, use_container_width=True)

            # 3D scatter if we have enough variation
            st.subheader("3D Yield Surface")
            fig_3d = px.scatter_3d(
                results_df,
                x="Fertilizer_per_Area",
                y="Annual_Rainfall",
                z="Predicted_Yield",
                color="Predicted_Yield",
                title="Yield vs Fertilizer & Rainfall",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Note**: The robot action recommendations (inside the box) are conceptual and expected 
    steps but are not experimentally verified because it's out of our current scope.
    
    *Robotic Decision Support System - Phase II*
    """)


if __name__ == "__main__":
    main()
