import os
import pickle

import streamlit as st
import pandas as pd
import plotly.express as px
import clean_data as cd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from model import trainmodel

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#2563eb",
    "#14b8a6",
    "#f59e0b",
    "#ef4444",
]

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Phishing Detection System",
    page_icon=":shield:",
    layout="wide"
)


# ==========================================================
# CSS STYLE
# ==========================================================

st.markdown(
    """
<style>

:root {
    --bg: #f6f8fb;
    --panel: #ffffff;
    --panel-soft: #f9fbff;
    --ink: #111827;
    --muted: #667085;
    --line: #d7deea;
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --primary-soft: #eaf1ff;
    --success: #0f766e;
    --warning: #b45309;
    --danger: #dc2626;
    --sidebar: #111827;
    --sidebar-soft: #1f2937;
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(37, 99, 235, 0.11), transparent 34rem),
        radial-gradient(circle at 88% 8%, rgba(20, 184, 166, 0.10), transparent 30rem),
        linear-gradient(180deg, #eef4ff 0, rgba(246, 248, 251, 0) 420px),
        var(--bg);
    color: var(--ink);
}

section.main > div {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}

h1, h2, h3 {
    letter-spacing: 0;
    color: var(--ink);
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}

hr {
    border-color: rgba(148, 163, 184, 0.22);
    margin: 1.5rem 0;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, var(--sidebar) 0%, #0f172a 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
    color: #eef2ff;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff;
    font-weight: 750;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111827;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label p {
    color: #cbd5e1;
    font-weight: 650;
}

.hero {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(135deg, rgba(17, 24, 39, 0.98), rgba(30, 64, 175, 0.94)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 80px);
    color: #ffffff;
    padding: 38px 40px;
    border-radius: 8px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, transparent 0 72%, rgba(20, 184, 166, 0.22) 72% 73%, transparent 73%),
        linear-gradient(0deg, transparent 0 58%, rgba(245, 158, 11, 0.18) 58% 59%, transparent 59%);
    pointer-events: none;
}

.eyebrow {
    text-transform: uppercase;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: #a7f3d0;
    margin-bottom: 12px;
}

.title {
    font-size: clamp(34px, 5vw, 54px);
    line-height: 1.03;
    font-weight: 850;
    color: #ffffff;
    max-width: 900px;
    position: relative;
    z-index: 1;
}


.subtitle {

    font-size: 18px;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.84);
    margin-top: 14px;
    max-width: 760px;
    position: relative;
    z-index: 1;

}

.hero-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 24px;
    position: relative;
    z-index: 1;
}

.hero-stat {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 8px;
    padding: 15px 16px;
    backdrop-filter: blur(10px);
}

.hero-stat strong {
    display: block;
    font-size: 18px;
    color: #ffffff;
}

.hero-stat span {
    display: block;
    margin-top: 4px;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.78);
}

.card {
    background: var(--panel);
    padding: 26px;
    border-radius: 8px;
    border: 1px solid var(--line);
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
    margin-bottom: 18px;
}

.card h2 {
    font-size: 23px;
    margin: 0 0 10px 0;
    color: #0f172a;
}

.card h3 {
    font-size: 16px;
    margin: 20px 0 8px 0;
}

.card p,
.card li {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.7;
}

.card ul {
    padding-left: 20px;
    margin: 8px 0 0 0;
}

.requirement-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 18px 0;
}

.requirement {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px;
    background: var(--panel-soft);
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
    box-shadow: inset 0 3px 0 var(--primary);
}

.requirement strong {
    display: block;
    color: var(--ink);
    margin-bottom: 4px;
}

.upload-card {
    border-left: 5px solid var(--success);
    background:
        linear-gradient(90deg, rgba(20, 184, 166, 0.08), transparent 38%),
        var(--panel);
}

table {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 8px;
    border: 1px solid var(--line);
    font-size: 14px;
}

th {
    background: #eef2f7;
    color: #1f2937;
    text-align: left;
    padding: 12px;
    font-weight: 750;
}

td {
    border-top: 1px solid var(--line);
    color: #374151;
    padding: 12px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    border: 1px solid var(--primary);
    background: linear-gradient(180deg, #3b82f6, var(--primary));
    color: #ffffff;
    font-weight: 750;
    min-height: 44px;
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.20);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--primary-dark);
    background: var(--primary-dark);
    color: #ffffff;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    border: 1px dashed #94a3b8;
    border-radius: 8px;
    background: #ffffff;
    padding: 16px;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

[data-testid="stMetricLabel"] p {
    color: var(--muted);
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: #0f172a;
    font-weight: 800;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--line);
}

[data-testid="stAlert"] {
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    border-radius: 8px;
}

.stSlider [data-baseweb="slider"] div {
    color: var(--primary);
}

@media (max-width: 800px) {
    .hero,
    .card {
        padding: 20px;
    }

    .hero-metrics,
    .requirement-grid {
        grid-template-columns: 1fr;
    }
}

</style>

""",
unsafe_allow_html=True
)



# ==========================================================
# HEADER
# ==========================================================


st.markdown(
    """
<div class="hero">
    <div class="eyebrow">Cybersecurity model lab</div>
    <div class="title">AI Email Phishing Detection System</div>
    <p class="subtitle">Upload labelled email datasets, clean noisy content, train NLP classifiers, and compare the signals that matter for phishing risk.</p>
    <div class="hero-metrics">
        <div class="hero-stat"><strong>3 models</strong><span>Logistic Regression, Random Forest, and SVM</span></div>
        <div class="hero-stat"><strong>CSV training</strong><span>Merge one or more labelled email datasets</span></div>
        <div class="hero-stat"><strong>Model export</strong><span>Download trained packages after evaluation</span></div>
    </div>
</div>
    """,
    unsafe_allow_html=True
)



st.divider()


st.markdown(
    """
<div class="card">

<h2>Workflow</h2>

<p>
Use this workspace to train machine learning models that classify emails as legitimate or phishing.
Upload one or more CSV files, choose models and charts from the sidebar, then click
<strong>Train Model</strong>.
</p>

<h3>CSV File Requirements</h3>

<div class="requirement-grid">
    <div class="requirement">
        <strong>body</strong>
        Required email message text.
    </div>
    <div class="requirement">
        <strong>label</strong>
        Required numeric class: 0 legitimate, 1 phishing.
    </div>
    <div class="requirement">
        <strong>subject</strong>
        Optional subject line, combined with the body when present.
    </div>
</div>

<ul>
    <li>Labels must be numeric. Empty labels and non-numeric labels are removed during cleaning.</li>
    <li>The cleaned dataset must contain both classes, with at least two rows for each class.</li>
</ul>

<h3>Example CSV Format</h3>

<table>
    <thead>
        <tr>
            <th>subject</th>
            <th>body</th>
            <th>label</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Account update</td>
            <td>Please review your account settings.</td>
            <td>0</td>
        </tr>
        <tr>
            <td>Urgent verification</td>
            <td>Click this link to verify your password now.</td>
            <td>1</td>
        </tr>
    </tbody>
</table>

</div>
""",
    unsafe_allow_html=True
)


st.divider()



# ==========================================================
# DATA FROM EXPERIMENT
# ==========================================================
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

if "confusion_matrices" not in st.session_state:
    st.session_state.confusion_matrices = {}

if "trained" not in st.session_state:
    st.session_state.trained = False

if "label_counts" not in st.session_state:
    st.session_state.label_counts = {}

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "trained_vectorizer" not in st.session_state:
    st.session_state.trained_vectorizer = None

if "is_training" not in st.session_state:
    st.session_state.is_training = False

if "training_requested" not in st.session_state:
    st.session_state.training_requested = False


def request_training():
    st.session_state.is_training = True
    st.session_state.training_requested = True


def clear_training_state():
    st.session_state.is_training = False
    st.session_state.training_requested = False







with st.sidebar:


    st.header("Model Settings")


    selected_models = st.multiselect(

        "Select Models",

        [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine"
        ],

        default=[

            "Random Forest"

        ]

    )


    st.divider()



    charts = st.multiselect(

        "Select Visualisation",

        [

            "Bar Chart",

            "Pie Chart",

            "Comparison Graph"

        ],

        default=[

            "Bar Chart"

        ]

    )


    st.divider()



    confidence = st.slider(

        "Detection Threshold",

        0.0,

        1.0,

        0.5

    )





st.markdown(
    """
<div class="card upload-card">

<h2>Upload Email Datasets</h2>

<p>Import CSV files with labelled email content. The app will combine, clean, vectorize, and train from the uploaded rows.</p>

</div>

""",
    unsafe_allow_html=True
)



files = st.file_uploader(

"Upload Files",

type=[

"csv"

],

accept_multiple_files=True

)
showresult: bool=False



if files:


    st.success(

        f"{len(files)} files uploaded"

    )


    upload_table=pd.DataFrame(

    {

    "File":

    [

    f.name for f in files

    ],

    "Size":

    [

    f.size for f in files

    ]

    }

    )


    st.dataframe(

        upload_table,

        width="stretch"

    )
    col1, col2 = st.columns(2)
    with col1:
        train_button = st.button(

            "Train Model",

            width="stretch",
            disabled=st.session_state.is_training,
            on_click=request_training

        )


    if st.session_state.training_requested:
        status = st.empty()

        datasets = []

        with st.spinner("Reading CSV files..."):

            for file in files:
                try:
                    df = pd.read_csv(file, low_memory=False)

                    required_columns = [
                        "body",
                        "label"
                    ]

                    missing = [
                        col for col in required_columns
                        if col not in df.columns
                    ]

                    if missing:
                        st.error(
                            f"{file.name} is missing columns: {missing}"
                        )
                        continue

                    # Add valid dataframe
                    datasets.append(df)

                except Exception as e:
                    st.error(
                        f"Error reading {file.name}: {e}"
                    )

        if datasets:

            with st.spinner("Combining datasets..."):

                combined = pd.concat(
                    datasets,
                    ignore_index=True
                )

            status.info(
                f"Combined dataset size: {combined.shape}"
            )



            cleandata = cd.CleanData(combined)
            cleaned_data = cleandata.clean()
            if cleaned_data['cleaned']:
                st.session_state.label_counts = cleaned_data["label_count"].to_dict()
                label_counts = cleaned_data["data"]["label"].value_counts()

                if label_counts.size < 2:
                    st.error(
                        "Training requires both classes in the cleaned data. "
                        f"Found labels: {label_counts.to_dict()}"
                    )
                    clear_training_state()
                    st.stop()

                if label_counts.min() < 2:
                    st.error(
                        "Training requires at least 2 rows for each class when using a stratified split. "
                        f"Found labels: {label_counts.to_dict()}"
                    )
                    clear_training_state()
                    st.stop()

                with st.spinner("TF-IDF Vectorization..."):
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X = vectorizer.fit_transform(cleaned_data["data"]["clean_email"])
                    y = cleaned_data["data"]["label"]
                    st.session_state.trained_vectorizer = vectorizer
                # Split the data into training and testing sets
                with st.spinner("Splitting data into training and testing sets..."):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )
                    except ValueError as e:
                        st.error(f"Unable to split the data for training: {e}")
                        clear_training_state()
                        st.stop()
                st.empty().write("Split the data into training and testing sets")
                with st.spinner("Training models..."):
                    trainmodel_class=trainmodel(X_train,y_train,X_test,y_test)
                    st.session_state.model_results = {}
                    st.session_state.confusion_matrices = {}
                    st.session_state.trained_models = {}

                    for model_name in selected_models:
                        st.write(f"Training and evaluating {model_name}")

                        try:
                            if model_name == "Random Forest":
                                rf_model = trainmodel_class.train_Random_Forest()
                                st.session_state.trained_models["Random Forest"] = rf_model
                                st.session_state.model_results["Random Forest"] = (
                                    trainmodel_class.evaluate_model(rf_model)
                                )
                                # Train Random Forest

                            elif model_name == "Support Vector Machine":
                                svm_model = trainmodel_class.train_Support_Vector_Machine()
                                st.session_state.trained_models["Support Vector Machine"] = svm_model
                                st.session_state.model_results["Support Vector Machine"] = (
                                    trainmodel_class.evaluate_model(svm_model)
                                )
                                # Train SVM

                            elif model_name == "Logistic Regression":
                                lr_model = trainmodel_class.train_Logistic_Regression()
                                st.session_state.trained_models["Logistic Regression"] = lr_model
                                st.session_state.model_results["Logistic Regression"] = (
                                    trainmodel_class.evaluate_model(lr_model)
                                )
                            else:
                                st.error(f"Unsupported model: {model_name}")
                        except ValueError as e:
                            st.error(f"{model_name} failed to train: {e}")
                    for model_name, results in st.session_state.model_results.items():
                        st.session_state.confusion_matrices[model_name] = results["conf_matrix"]


                    if st.session_state.model_results:
                        st.success("Models trained and evaluated  successfully!")
                        showresult=True
                    else:
                        st.error("No models were trained successfully.")
            else:
                st.error("Data cleaning failed. Please check the logs for details.")


        else:
            st.warning(
                "No valid CSV files found. Make sure files contain body and label columns."
            )

        clear_training_state()








model_results_df = pd.DataFrame(
    [
        {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"]
        }

        for model_name, metrics in st.session_state.model_results.items()
    ]
)

st.divider()
if not model_results_df.empty:
    trained_selected_models = [
        model for model in selected_models
        if model in st.session_state.model_results
    ]

    st.header("Model Performance")



    if not selected_models:


        st.warning(

            "Select a model from sidebar"

        )


    else:


        selected_data = model_results_df[model_results_df["Model"].isin(trained_selected_models)]
        if selected_data.empty:
            st.warning("None of the selected models have training results yet.")
        st.dataframe(

            selected_data,

            width="stretch"

        )

        if trained_selected_models and st.session_state.trained_vectorizer is not None:
            st.subheader("Download Trained Model")
            download_model_name = st.selectbox(
                "Choose model to download",
                trained_selected_models,
                key="download_model_name"
            )
            model_package = {
                "model_name": download_model_name,
                "model": st.session_state.trained_models[download_model_name],
                "vectorizer": st.session_state.trained_vectorizer,
                "labels": {
                    0: "Legitimate",
                    1: "Phishing"
                }
            }
            model_bytes = pickle.dumps(model_package)
            file_name = download_model_name.lower().replace(" ", "_") + "_phishing_model.pkl"
            st.download_button(
                "Download Model",
                data=model_bytes,
                file_name=file_name,
                mime="application/octet-stream",
                width="stretch"
            )



        # Metrics


        for _, row in selected_data.iterrows():


            st.subheader(

                row["Model"]

            )


            a,b,c,d = st.columns(4)



            a.metric(

                "Accuracy",

                f"{row['Accuracy']*100:.2f}%"

            )


            b.metric(

                "Precision",

                f"{row['Precision']*100:.2f}%"

            )


            c.metric(

                "Recall",

                f"{row['Recall']*100:.2f}%"

            )


            d.metric(

                "F1 Score",

                f"{row['F1 Score']*100:.2f}%"

            )





    st.divider()


    st.header("Visual Analytics")



    if trained_selected_models:

        selected_data = model_results_df[model_results_df["Model"].isin(trained_selected_models)]
        chart_data = selected_data



        if "Bar Chart" in charts:


            fig = px.bar(

                chart_data,

                x="Model",

                y="Accuracy",

                text="Accuracy",

                title="Accuracy Comparison",

                color_discrete_sequence=["#2563eb"]

            )


            st.plotly_chart(

                fig,

                width="stretch"

            )





        if "Comparison Graph" in charts:


            melted = chart_data.melt(

                id_vars="Model",

                value_vars=[

                    "Accuracy",

                    "Precision",

                    "Recall",

                    "F1 Score"

                ]

            )



            fig = px.line(

                melted,

                x="Model",

                y="value",

                color="variable",

                markers=True,

                title="Model Metric Comparison"

            )



            st.plotly_chart(

                fig,

                width="stretch"

            )




        if "Pie Chart" in charts:

            legitimate_count = st.session_state.label_counts.get(0, 0)
            phishing_count = st.session_state.label_counts.get(1, 0)
            dataset = pd.DataFrame(

            {

            "Class":

            [

            "Legitimate",

            "Phishing"

            ],

            "Count":

            [

                legitimate_count,
             phishing_count

            ]

            }

            )



            fig = px.pie(

                dataset,

                names="Class",

                values="Count",

                title="Dataset Distribution",

                color="Class",

                color_discrete_map={
                    "Legitimate": "#0f766e",
                    "Phishing": "#dc2626"
                }

            )


            st.plotly_chart(

                fig,

                width="stretch"

            )





    st.divider()


    st.header("Confusion Matrix Analysis")



    if trained_selected_models:


        for model in trained_selected_models:


            st.subheader(model)



            cm = st.session_state.confusion_matrices[model]



            cm_df = pd.DataFrame(

                cm,

                columns=[

                    "Predicted Legitimate",

                    "Predicted Phishing"

                ],

                index=[

                    "Actual Legitimate",

                    "Actual Phishing"

                ]

            )



            st.table(cm_df)




            TN = cm[0][0]

            FP = cm[0][1]

            FN = cm[1][0]

            TP = cm[1][1]




            a,b,c,d = st.columns(4)


            a.metric(

                "True Positive",

                TP

            )


            b.metric(

                "True Negative",

                TN

            )


            c.metric(

                "False Positive",

                FP

            )


            d.metric(

                "False Negative",

                FN

            )




            fnr = FN/(FN+TP) if (FN+TP) else 0



            st.error(

                f"False Negative Rate: {fnr:.2%}"

            )



            heatmap = px.imshow(

                cm_df,

                text_auto=True,

                title=f"{model} Confusion Matrix",

                color_continuous_scale=[
                    "#f8fafc",
                    "#93c5fd",
                    "#1d4ed8"
                ]

            )



            st.plotly_chart(

                heatmap,

                width="stretch"

            )





st.divider()


st.caption(

"""
MSc Research Project:

An Analytical Evaluation of Machine Learning Techniques for Email Phishing Detection
to Enhance Cybersecurity Risk Management and Organisational Resilience

"""

)
