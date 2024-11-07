from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

import pandas as pd
import streamlit as st

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Datasets", page_icon="📊")

datasets = automl.registry.list(type="dataset")

st.write("# 📊 Datasets")
st.write(
    "Saved datasets in the system:",
    ", ".join(dataset.name for dataset in datasets),
)

# Upload a new dataset in CSV format
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

if csv_file is not None:
    dataframe = pd.read_csv(csv_file)
    file_name = csv_file.name

    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    st.write(dataframe.head())

    dataset = Dataset.from_dataframe(data=dataframe, name=file_name, asset_path=file_name)

    if dataset not in datasets:
        automl._storage.save(dataset.save(dataframe), file_name)
        automl._registry.register(dataset)

    st.write("Dataset successfully uploaded. Head to the modeling page to use it.")
