import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

if "executed_pipeline" in st.session_state:
    st.session_state.result = None
    st.session_state.executed_pipeline = None

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Datasets", page_icon="📊")

datasets = automl.registry.list(type="dataset")

st.write("# 📊 Datasets")
st.write("Currently saved datasets:", ", ".join(dataset.name
                                                for dataset in datasets))

csv_file = st.file_uploader("Upload your own csv dataset", ["csv"])

if csv_file is not None:
    dataframe = pd.read_csv(csv_file)
    file_name = csv_file.name

    shuffle_box = st.checkbox("Would you like to shuffle the data?",
                              value=False)
    if shuffle_box:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    st.write(dataframe.head())

    tags_input = st.text_input("Enter tags for this dataset "
                               "(comma-separated):", "")
    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

    dataset = Dataset.from_dataframe(dataframe, file_name, file_name,
                                     tags=tags)

    save_btn = st.button("Save Dataset")
    if save_btn:
        if dataset not in datasets:
            automl._storage.save(dataset.save_df(dataframe), file_name)
            automl._registry.register(dataset)
            if len(tags) > 0:
                st.success(f"{file_name} saved to datasets with tags: "
                           f"{', '.join(tags)}!")
            else:
                st.success(f"{file_name} saved to datasets!")
        else:
            st.warning(f"{file_name} is already saved in datasets.")

    st.write("Head to the modelling page once the dataset is saved!")
