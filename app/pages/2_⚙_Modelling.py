import io
from math import ceil

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    get_metric,
)
from autoop.core.ml.model import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    get_model,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

MIN_TRAINING_SAMPLES = 3

st.set_page_config(page_title="Modelling", page_icon="📈")


def display_helper_text(text: str) -> None:
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


# Initialize session state variables
for key in ["result", "executed_pipeline", "active_pipeline", "result_data"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.write("# ⚙ Modelling")
display_helper_text(
    "In this section, you can design a " + "machine learning pipeline to "
                                           "train a model on a dataset."
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")
pipelines = automl.registry.list(type="pipeline")

analysis_metrics = []
model_chosen, features_chosen, metrics_chosen = False, False, False

st.write("## Select Dataset:")
dataset_name = st.selectbox(
    "Choose dataset to apply to model or upload your own from "
    "the datasets page:",
    (ds.name for ds in datasets),
    index=None,
)

if dataset_name:
    st.write("## Remove Dataset")
    if st.button("Remove dataset"):
        for ds in datasets:
            if ds.name == dataset_name:
                automl.registry.delete(ds.id)
                st.rerun()

    selected_data = next(ds for ds in datasets if ds.name == dataset_name)
    data_content = selected_data.data.decode()
    loaded_data = pd.read_csv(io.StringIO(data_content))
    st.write("Selected Data:", loaded_data.head())

    updated_dataset = Dataset.from_dataframe(
        name=selected_data.name,
        data=loaded_data,
        asset_path=selected_data.asset_path,
        version=selected_data.version,
    )
    feature_list = detect_feature_types(updated_dataset)
    st.write("## Feature Selection:")

    target_column = st.selectbox(
        "Select target column for prediction:", feature_list, index=None
    )
    if target_column:
        input_columns = [
            f for f in feature_list if f.name != target_column.name
        ]
        st.write(f"Target column: {target_column.name}")

        selected_inputs = st.multiselect(
            "Choose input columns for model:", input_columns
        )
        if selected_inputs:
            features_chosen = True
            st.write(
                "Selected Columns:",
                ", ".join(feat.name for feat in selected_inputs),
            )

        target_type = target_column.type
        st.write("## Model Selection:")

        if target_type == "numerical":
            st.write("Detected task: Regression")
            selected_model = st.selectbox(
                "Choose model to use:", REGRESSION_MODELS
            )
            model_instance = get_model(selected_model)
            if model_instance:
                model_chosen = True

            analysis_metrics = [
                get_metric(m)
                for m in st.multiselect("Select metrics:", REGRESSION_METRICS)
            ]
            metrics_chosen = bool(analysis_metrics)

        elif target_type == "categorical":
            st.write("Detected task: Classification")
            selected_model = st.selectbox(
                "Choose model to use:", CLASSIFICATION_MODELS
            )
            model_instance = get_model(selected_model)
            if model_instance:
                model_chosen = True

            analysis_metrics = [
                get_metric(m)
                for m in st.multiselect(
                    "Select metrics:", CLASSIFICATION_METRICS
                )
            ]
            metrics_chosen = bool(analysis_metrics)

if model_chosen and metrics_chosen and features_chosen:
    min_ratio = ceil(MIN_TRAINING_SAMPLES / len(loaded_data) * 100) / 100
    train_ratio = st.slider(
        "Select proportion of data for training:", min_ratio, 0.99, 0.80
    )

    model_pipeline = Pipeline(
        metrics=analysis_metrics,
        dataset=updated_dataset,
        model=model_instance,
        input_features=selected_inputs,
        target_feature=target_column,
        split=train_ratio,
    )

    st.write("## Pipeline Summary:")
    st.write("- **Dataset**:", updated_dataset.name)
    st.write(f"- **Tags:** {', '.join(selected_data.tags)}")
    st.write("- **Target Column**:", target_column.name)
    st.write(
        "- **Input Columns**:",
        ", ".join(feat.name for feat in selected_inputs),
    )
    st.write("- **Model**:", model_instance.__class__.__name__)
    st.write(
        "- **Metrics**:",
        ", ".join(metric.__class__.__name__ for metric in analysis_metrics),
    )
    st.write("- **Training Ratio**:", f"{train_ratio:.0%} of data")

    max_predictions = st.number_input(
        "Specify max predictions to display (0=all):",
        min_value=0,
        value=50,
        step=1,
    )

    if st.button("Run Pipeline"):
        st.session_state.result_data = model_pipeline.execute()
        st.session_state.active_pipeline = model_pipeline

    if st.session_state.active_pipeline:
        pipeline_output = st.session_state.result_data
        train_output = pipeline_output["train_metrics"]
        test_output = pipeline_output["test_metrics"]
        preds = pipeline_output["predictions"]

        if target_column.type == "categorical":
            unique_vals = loaded_data[target_column.name].unique()
            preds = [unique_vals[int(pred)] for pred in preds]

        preds_df = pd.DataFrame(preds, columns=[target_column.name])

        st.write("## Pipeline Results:")

        st.write("### Training Metrics:")
        for result in train_output:
            st.write(f"- **{result[0].__class__.__name__}**: {result[1]:.4f}")

        st.write("### Testing Metrics:")
        for result in test_output:
            st.write(f"- **{result[0].__class__.__name__}**: {result[1]:.4f}")

        st.write("### Predictions:")
        if max_predictions == 0 or max_predictions >= len(preds):
            st.dataframe(preds_df, use_container_width=True)
        else:
            st.dataframe(preds_df.head(max_predictions), use_container_width=True)
            st.write(f"... and {len(preds) - max_predictions} more.")
