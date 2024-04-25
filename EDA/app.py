import streamlit as st
import pandas as pd

# import tracemalloc
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    auc,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

import numpy as np
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance

# import pathlib
import uuid

# import objgraph
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import streamlit.components.v1 as components
from pandas_profiling import ProfileReport

from streamlit_javascript import st_javascript
import json


@st.cache_resource(ttl="12h", max_entries=10)
def load_unpkg(src: str) -> str:
    return requests.get(src).text


HTML_2_CANVAS = load_unpkg("https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.js")
HTML_2_CANVAS_MAP = load_unpkg(
    "https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.js.map"
)

# print(HTML_2_CANVAS)


@st.cache_resource(ttl="24h", max_entries=10)
def read_js_file(filename):
    with open(filename) as file:
        content = file.read()
    return content


# send_log_js = open("./scripts/sendLog.js")
# send_log_code = send_log_js.read()
send_log_code = read_js_file("./scripts/sendLog.js")

# image_save_btns_js = open("./scripts/prepareImageSaveBtns.js")
# image_save_btns_code = image_save_btns_js.read()
image_save_btns_code = read_js_file("./scripts/prepareImageSaveBtns.js")

# create_canvas_js = open("./scripts/createCanvas.js")
# create_canvas_code = create_canvas_js.read()
create_canvas_code = read_js_file("./scripts/createCanvas.js")

### --------------------------------------------- Capture query params --------------------------------------------- ###
if "url_params" not in st.session_state:
    url_params = st.experimental_get_query_params()
    st.session_state.url_params = url_params
else:
    url_params = st.session_state.url_params


### --------------------------------------------------- Functions def ---------------------------------------------- ###
# Image Manipulation
@st.cache_data(persist=True, max_entries=10)
def load_image(img):
    im = Image.open(os.path.join(img))
    return im


def getUniqueKey():
    return str(uuid.uuid4())


# Everytime a log is created, a 26px div is created which shows up as empty space
# this function will remove those divs
def remove_st_javsacript():
    components.html(
        f"""
        <script type='text/javascript'>
            const streamDocument = window.parent.document;
            const verticalBlock = streamDocument.querySelector('.main [data-testid="stVerticalBlock"]');

            const dataStales = verticalBlock.querySelectorAll('[data-stale]');
            for (const dataStale of dataStales) {{
                if(dataStale.querySelector('iframe')) {{
                    dataStale.style.display = "none";
                }}
            }}
        </script>
    """,
        width=0,
        height=0,
    )


# Send log file to flask server in fetch POST request
def send_log(action):
    if all([key in url_params for key in ["id", "enrollment_id", "problem_id"]]):
        log_content = {
            "profile_id": url_params["id"][0],
            "enrollment_id": url_params["enrollment_id"][0],
            "problem_id": url_params["problem_id"][0],
            "dataset": url_params["dataset"][0],
            "task": action,
            "type": "log",
        }

        components.html(
            f"""
                    <script type='text/javascript'>
                        {send_log_code}
                        sendLog({{...{json.dumps(log_content)}}})
                    </script>
                """,
            height=0,
            width=0,
        )

        remove_st_javsacript()

    else:
        print("Currently no query params, no code log sent.")


@st.cache_resource(ttl="12h", max_entries=10)
def show_html(HtmlFile):
    return HtmlFile.read()


@st.cache_resource(ttl="12h", max_entries=10)
def read_csv(DATA_URL) -> pd.DataFrame:
    """## Iris DataFrame
    Returns:
        pd.DataFrame -- A dataframe with the source iris data
    """
    return pd.read_csv(DATA_URL)


def user_input_features(features):
    data = {}
    for _, j in enumerate(features.columns):
        var0 = st.sidebar.slider(
            j,
            float(features[j].min()),
            float(features[j].max()),
            float(features[j].mean()),
        )
        # print(float(features[j].mean()))
        data[j] = var0

    features = pd.DataFrame(data, index=[0])
    clfs = []
    clfs = st.multiselect(
        "Select models",
        (
            "Random Forest",
            "Logistic Regression",
            "Support Vector",
            "Naive Bayes",
            "Decision Tree",
            "K Nearest Neighbour",
            "linear discriminant",
        ),
    )
    return features, clfs


@st.cache_data(max_entries=5, ttl=1800)
def compute_predition(clfs, features, labels, df_test):
    models = []
    predictions = {}
    if "Random Forest" in clfs:
        models.append(("RF", RandomForestClassifier()))
    if "Logistic Regression" in clfs:
        models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
    if "Support Vector" in clfs:
        models.append(("SVM", SVC(gamma="auto", probability=True)))
    if "Naive Bayes" in clfs:
        models.append(("NB", GaussianNB()))
    if "Decision Tree" in clfs:
        models.append(("CART", DecisionTreeClassifier()))
    if "K Nearest Neighbour" in clfs:
        models.append(("KNN", KNeighborsClassifier()))
    if "linear discriminant" in clfs:
        models.append(("LDA", LinearDiscriminantAnalysis()))

    # Train on all data
    # X, x_test, Y, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

    for name, model in models:
        model.fit(features, labels)
        prediction = model.predict(df_test)
        prediction_proba = model.predict_proba(df_test)
        predictions[name] = (prediction_proba[0], prediction[0])

    return predictions


def show_scatter_plot(selected_species_df: pd.DataFrame):
    """## Component to show a scatter plot of two features for the selected species

    Arguments:
        selected_species_df {pd.DataFrame} -- A DataFrame with the same columns as the
            source_df iris dataframe
    """
    st.subheader("Scatter plot")
    feature_x = st.selectbox("Which feature on x?", selected_species_df.columns[0:-1])
    feature_y = st.selectbox("Which feature on y?", selected_species_df.columns[0:-1])

    fig = px.scatter(
        selected_species_df,
        x=feature_x,
        y=feature_y,
        color=selected_species_df.columns[-1],
    )
    st.plotly_chart(fig)


def select_species(source_df: pd.DataFrame) -> pd.DataFrame:
    """## Component for selecting one of more species for exploration

    Arguments:
        source_df {pd.DataFrame} -- The source iris dataframe

    Returns:
        pd.DataFrame -- A sub dataframe having data for the selected species
    """
    selected_species = st.sidebar.multiselect(
        "Scatter plot (select class instances):",
        source_df.iloc[:, -1].unique(),
    )
    # selected_species_df = source_df[(source_df["variety"].isin(selected_species))]
    selected_species_df = source_df[(source_df.iloc[:, -1].isin(selected_species))]
    if selected_species:
        st.write(selected_species_df)
    return selected_species_df


def show_histogram_plot(selected_species_df: pd.DataFrame):
    """## Component to show a histogram of the selected species and a selected feature

    Arguments:
        selected_species_df {pd.DataFrame} -- A DataFrame with the same columns as the
            source_df iris dataframe
    """
    st.subheader("Histogram")
    feature = st.selectbox("Which feature?", selected_species_df.columns[0:-1])
    fig2 = px.histogram(
        selected_species_df,
        x=feature,
        color=selected_species_df.columns[-1],
        marginal="rug",
    )
    st.plotly_chart(fig2)


def _handle_missing(features, labels):
    appraoch = st.sidebar.radio(
        "Handle missing data",
        ("impute with mean", "Replace with zero", "drop the entry"),
    )
    if appraoch == "impute with mean":
        # approach 1 (imputing with the mean)
        new_features = features.fillna(features.mean())
        new_labels = labels.fillna(labels.mean())
    elif appraoch == "drop the entry":
        # approach 2 (delete the row of the nan values)
        # TODO consider changing the output size as well
        dum = pd.concat([features, labels], axis=1, join="inner").dropna(axis=0)
        new_features = dum.iloc[:, :-1]
        new_labels = dum.iloc[:, -1]
        del dum
    else:
        # approach 3 (replace it with zero)
        new_features = features.fillna(0)
        new_labels = labels.fillna(0)
    return new_features, new_labels


def handle_io(source_df):
    features = source_df.iloc[:, :-1]
    # check if all the colomn are numeric
    # for col in features.columns:
    #     if not pd.api.types.is_numeric_dtype(features[col]):
    #         st.warning(
    #             "Colomn <{}> is not numeric. We have mapped it to numeric values.".format(
    #                 col
    #             )
    #         )
    #         n = features.groupby(col).ngroups
    #         Map = {
    #             key: index / (n - 1)
    #             for index, key in enumerate(features.groupby(col).groups.keys())
    #         }
    #         features[col] = features[col].map(Map, na_action=None).astype(float)
    cols = []
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            cols.append(col)
            n = features.groupby(col).ngroups
            Map = {
                key: index / (n - 1)
                for index, key in enumerate(features.groupby(col).groups.keys())
            }
            features[col] = features[col].map(Map, na_action=None).astype(float)
    if cols:
        st.warning(
            "Colomn {} is/are not numeric. We have mapped it to numeric values.".format(
                cols
            )
        )

    labels = source_df.iloc[:, -1]
    labels = source_df.iloc[:, -1]
    # check if all the colomn are numeric
    # if not pd.api.types.is_numeric_dtype(labels):
    #     print('bye')
    #     st.warning("Output value is not numeric. We have mapped it to numeric values.")
    #     Map = {key: index/(labels.groupby(level=0).ngroups-1) for index, key in enumerate(labels.groupby(level=0).groups.keys())}
    #     labels = labels.map(Map, na_action=None).astype(float)

    if labels.isnull().values.any() or features.isnull().values.any():
        st.warning(
            "There are null values in your dataset. Please choose a method in the side bar to handle it."
        )
        features, labels = _handle_missing(features, labels)

    return features, labels


def show_machine_learning_model(source_df: pd.DataFrame):
    """Component to show the performance of an ML Algo trained on the iris data set

    Arguments:
        source_df {pd.DataFrame} -- The source iris data set

    Raises:
        NotImplementedError: Raised if a not supported model is selected
    """
    st.header("Machine Learning models")

    # Handle missing data and non-numeric values
    features, labels = handle_io(source_df)

    ratio = st.sidebar.slider("Train-test ratio", 0.0, 1.0, 0.7)
    sampling = st.sidebar.radio(
        "Sampling method",
        (
            "None",
            "class weight",
            "Random under sampler",
            "Random over sampler",
            "SMOTE",
        ),
    )
    # if sampling != "None":
    #     st.info("TODO: to be completed!")

    # x_train, x_test, y_train, y_test = train_test_split(
    #     features, labels, train_size=ratio, random_state=1
    # )

    # alg = [
    #     "Logistic Regression",
    #     "Random Forest",
    #     "Decision Tree",
    #     "Support Vector Machine",
    #     "Naive Bayes",
    #     "K Nearest Neighbour",
    #     "linear discriminant",
    # ]
    # classifier = st.selectbox("Which algorithm?", alg)

    # send_log(f"{dataset}/modelling/{classifier}")

    # if classifier == "Random Forest":
    #     model = RandomForestClassifier()
    # elif classifier == "Logistic Regression":
    #     model = LogisticRegression(solver="liblinear", multi_class="ovr")
    # elif classifier == "Support Vector Machine":
    #     model = SVC(gamma="auto", probability=True)
    # elif classifier == "Naive Bayes":
    #     model = GaussianNB()
    # elif classifier == "Decision Tree":
    #     model = DecisionTreeClassifier()
    # elif classifier == "K Nearest Neighbour":
    #     model = KNeighborsClassifier()
    # elif classifier == "linear discriminant":
    #     model = LinearDiscriminantAnalysis()
    # else:
    #     raise NotImplementedError()

    if sampling == "class weight":
        class_weight = "balanced"
    else:
        class_weight = None

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=ratio, random_state=1
    )

    alg = [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "Support Vector Machine",
        "Naive Bayes",
        "K Nearest Neighbour",
        "linear discriminant",
    ]
    classifier = st.selectbox("Which algorithm?", alg)

    if (
        classifier in ["Naive Bayes", "K Nearest Neighbour", "linear discriminant"]
        and sampling == "class weight"
    ):
        st.error(
            "Classifier '{}' can not have 'class weight' as a sampling method!".format(
                classifier
            )
        )

    if classifier == "Random Forest":
        model = RandomForestClassifier(class_weight=class_weight)
    elif classifier == "Logistic Regression":
        model = LogisticRegression(
            random_state=0,
            solver="liblinear",
            multi_class="ovr",
            class_weight=class_weight,
        )
    elif classifier == "Support Vector Machine":
        model = SVC(gamma="auto", probability=True, class_weight=class_weight)
    elif classifier == "Naive Bayes":
        model = GaussianNB()
    elif classifier == "Decision Tree":
        model = DecisionTreeClassifier(class_weight=class_weight)
    elif classifier == "K Nearest Neighbour":
        model = KNeighborsClassifier()
    elif classifier == "linear discriminant":
        model = LinearDiscriminantAnalysis()
    else:
        raise NotImplementedError()

    smp = None
    if sampling == "Random under sampler":
        smp = RandomUnderSampler()
    elif sampling == "Random over sampler":
        smp = RandomOverSampler(sampling_strategy="auto")
    elif sampling == "SMOTE":
        smp = SMOTE(random_state=27, sampling_strategy="minority", k_neighbors=5)

    if smp:
        x_train, y_train = smp.fit_resample(x_train, y_train)

    model.fit(x_train, y_train)

    pred_model = model.predict(x_test)
    cm_model = confusion_matrix(y_test, pred_model)

    classes = source_df.iloc[:, -1].unique()

    ytest = label_binarize(y_test, classes=classes)
    ypred = label_binarize(pred_model, classes=classes)

    d = {
        "Accuracy": model.score(x_test, y_test).round(4),
        "AUC score": roc_auc_score(ytest, ypred).round(4),
        "Precision": precision_score(ytest, ypred, average="micro").round(4),
        "Recall": recall_score(ytest, ypred, average="micro").round(4),
        "F1 score": f1_score(ytest, ypred, average="micro").round(4),
    }
    st.write("Metrics ")
    st.table(pd.DataFrame(data=d, index=[0]))

    st.write("Confusion matrix ", cm_model)
    st.write("ROC curve ")

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(ytest.shape[-1]):
        fpr[i], tpr[i], _ = roc_curve(ytest[:, i], ypred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ytest.ravel(), ypred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
    )
    for i in range(ytest.shape[-1]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("micro-average ROC curve")
    plt.legend(loc="lower right")
    st.pyplot()


def delete_session_key(target):
    for key in st.session_state.keys():
        if key == target:
            del st.session_state[key]
            return


def create_canvas(temp_key, type):
    if all([key in url_params for key in ["id", "enrollment_id", "problem_id"]]):
        log_content = {
            "profile_id": url_params["id"][0],
            "enrollment_id": url_params["enrollment_id"][0],
            "problem_id": url_params["problem_id"][0],
            "type": "image",
        }

        # Convert figure type to proper document selector
        if type == "dataframe":
            query = ".stDataFrame"
        elif type == "image":
            query = "[data-testid='stImage']"
        elif type == "vega_plot":
            query = "[data-testid='stArrowVegaLiteChart']"
        elif type == "plotly_chart":
            query = ".stPlotlyChart"

        components.html(
            f"""
                    <script type='application/javascript'>{HTML_2_CANVAS}</script>
                    <script type='text/javascript'>
                        {create_canvas_code}
                        createCanvas("{temp_key}", "{type}", "{query}", {{...{json.dumps(log_content)}}})
                    </script>
                """,
            height=0,
            width=0,
        )
    else:
        print("No user signed in, therefore image cannot be saved.")


def update_button_label(temp_key, type):
    components.html(
        f"""
            <script type='text/javascript'>
                {image_save_btns_code}
                updateButtonLabel("{temp_key}", "{type}");
            </script>
            """,
        height=0,
        width=0,
    )


def setup_screenshot_button(temp_key, type="dataframe"):
    if all([key in url_params for key in ["id", "enrollment_id", "problem_id"]]):
        st.markdown("###### Add a title and save a screenshot for your notes")
        st.text_input(
            "Add title to screenshot",
            label_visibility="collapsed",
            placeholder="Optional title",
            key=temp_key,
        )
        save_image_btn = st.button(label=temp_key)
        update_button_label(temp_key, type)
        if save_image_btn:
            create_canvas(temp_key, type)
    else:
        print("No user signed in, screenshot button removed.")


def clear_session_state_if_new_task(task):
    if "task" not in st.session_state:
        st.session_state.task = task
    elif st.session_state.task != task:
        session_state_copy = dict(st.session_state)
        for key in session_state_copy:
            if key != "url_params":
                del st.session_state[key]
        st.session_state.task = task
        del session_state_copy


### --------------------------------------------------- MAIN ------------------------------------------------------- ###
st.title("DaTu EDA/MLAAS toolbox")

st.set_option("deprecation.showPyplotGlobalUse", False)

st.sidebar.header("User input")
if "dataset" in url_params:
    dataset = " ".join(url_params["dataset"][0].split("_"))
else:
    dataset = st.sidebar.radio(
        "Please select your dataset",
        ("Iris", "PIMA", "Wine", "Health Insurance", "External"),
    )
st.sidebar.markdown(f"**_{dataset}_** dataset loaded")
### --------------------------------------------------- Dataset ---------------------------------------------------- ###
DATA_URL = ""
if dataset == "Iris":
    DATA_URL = "https://raw.githubusercontent.com/datu-ca/dataset/main/classification/IRIS/data.csv"
elif dataset == "PIMA":
    DATA_URL = "https://raw.githubusercontent.com/datu-ca/dataset/main/classification/PIMA/data.csv"
elif dataset == "Wine":
    DATA_URL = "https://raw.githubusercontent.com/datu-ca/dataset/main/classification/WINE/data.csv"
elif dataset == "Health Insurance":
    DATA_URL = "https://raw.githubusercontent.com/datu-ca/dataset/main/classification/HEALTH/data.csv"
else:
    type = st.sidebar.radio("Please select", ("Upload", "URL"))
    if type == "Upload":
        DATA_URL = st.sidebar.file_uploader(
            "Choose CSV file", type=["csv"], accept_multiple_files=False
        )
    else:
        DATA_URL = st.sidebar.text_input("Input remote URL (raw format)")
if DATA_URL:
    try:
        source_df = read_csv(DATA_URL)
    except:
        st.error("The file/URL is not parsable.")
        st.warning("Please select another file/URL or choose the preloaded dataset.")
        DATA_URL = ""

### --------------------------------------------------- Task  ------------------------------------------------------ ###
task = ""
if DATA_URL:
    task = st.sidebar.radio(
        "Please select your task",
        ("Data review", "Visualization", "Modeling", "Prediction"),
    )

st.sidebar.write(
    """
---
"""
)

# if st.button("Run Memory Diagnostics"):
#     # Identify the leaking objects
#     objgraph.show_most_common_types()

#     objgraph.show_growth()

#     objgraph.show_chain(objgraph.by_type("ReferenceType"), filename="chain.png")

# Analyze the references (replace 'YourLeakingType' with the actual type)
# x = objgraph.by_type('YourLeakingType')
# objgraph.show_backrefs(x[:3], max_depth=5)

# if st.sidebar.button("Reset"):
#     st.session_state.clear()


# Function to display the top memory-consuming lines
# def display_top(snapshot, top_n=10):
#     st.write(f"Top {top_n} lines consuming memory:")
#     for stat in snapshot.statistics("lineno")[:top_n]:
#         st.write(stat)


# # Streamlit UI
# st.title("Memory Profiling with tracemalloc")

# Button to start tracing memory allocations
# if st.sidebar.button("Start Tracing"):
#     tracemalloc.start()
#     st.write("Started tracing memory allocations!")

# # Button to display memory allocation results
# if st.sidebar.button("Show Results"):
#     if tracemalloc.is_tracing():
#         snapshot = tracemalloc.take_snapshot()
#         display_top(snapshot)
#     else:
#         st.write("Please start tracing first!")


clear_session_state_if_new_task(task)

### --------------------------------------------------- Data review  ----------------------------------------------- ###
if task == "Data review":
    log_action = f"{dataset}/Data_review"
    # send_log(log_action)

    # Show Dataset
    if st.checkbox("Preview DataFrame"):
        depth1 = log_action + "/preview_dataFrame"
        send_log(depth1)
        head_btn = st.button("Head")
        if head_btn or "head" in st.session_state:
            temp_key = depth1 + "/head"
            st.write(source_df.head())
            send_log(temp_key)
            setup_screenshot_button(temp_key, "dataframe")
            st.session_state.head = True
        tail_btn = st.button("Tail")
        if tail_btn or "tail" in st.session_state:
            temp_key = depth1 + "/tail"
            st.write(source_df.tail())
            send_log(temp_key)
            setup_screenshot_button(temp_key, "dataframe")
            st.session_state.tail = True

    # Show Entire Dataframe
    if st.checkbox("Show All DataFrame") or "showAll" in st.session_state:
        temp_key = log_action + "/show_all_dataFrame"
        st.dataframe(source_df)
        send_log(temp_key)
        setup_screenshot_button(temp_key, "dataframe")
        st.session_state.showAll = True

    # Dimensions
    data_dim = st.radio("What Dimension Do You Want to Show", ("Rows", "Columns"))
    if data_dim == "Rows" or "rows" in st.session_state:
        send_log(log_action + "/rows")
        st.text("Number of data points")
        st.write(len(source_df))
        delete_session_key("columns")
        st.session_state.rows = True
    if data_dim == "Columns" or "columns" in st.session_state:
        temp_key = log_action + "/columns"
        st.text("Number of features")
        st.write(source_df.shape[1])
        st.write(source_df.columns)
        send_log(temp_key)
        setup_screenshot_button(temp_key)
        delete_session_key("rows")
        st.session_state.columns = True

    if st.checkbox("Show Summary of Dataset") or "showSummary" in st.session_state:
        temp_key = log_action + "/show_summary_of_dataset"
        st.write(source_df.describe())
        send_log(temp_key)
        setup_screenshot_button(temp_key, "dataframe")
        st.session_state.showSummary = True

    if st.checkbox("Categories"):
        send_log(log_action + "/categories")
        st.write(source_df.groupby(source_df.columns[-1]).size())

    if st.checkbox("Report (This might take a while for heavy datasets)"):
        depth1 = log_action + "/report"
        if dataset == "External":
            send_log(depth1 + "/external")
            file = ProfileReport(source_df)
            file.to_file(output_file="output/{}.html".format(dataset))
        try:
            with open(
                "output/{}.html".format(dataset), "r", encoding="utf-8"
            ) as HtmlFile:
                source_code = show_html(HtmlFile)
        except:
            file = ProfileReport(source_df)
            file.to_file(output_file="output/{}.html".format(dataset))
            with open(
                "output/{}.html".format(dataset), "r", encoding="utf-8"
            ) as HtmlFile:
                source_code = show_html(HtmlFile)
        components.html(source_code, height=600, scrolling=True)


### --------------------------------------------------- Visualization  --------------------------------------------- ###
if task == "Visualization":
    log_action = f"{dataset}/visualization"
    send_log(log_action)

    selected_species_df = select_species(source_df)
    if not selected_species_df.empty:
        show_scatter_plot(selected_species_df)
        send_log(log_action + "/scatter_plot")
        setup_screenshot_button(log_action + "/scatter_plot", "plotly_chart")

        show_histogram_plot(selected_species_df)
        send_log(log_action + "/histogram_plot")
        setup_screenshot_button(log_action + "/histogram_plot", "plotly_chart")

    if st.sidebar.checkbox("Group plots"):
        depth1 = log_action + "/group_plots"
        send_log(depth1)
        # Show Plots
        if (
            st.checkbox(
                "Plot with Matplotlib (For large dataset, this might take a while) "
            )
            # or "matplotlib" in st.session_state
        ):
            temp_key = depth1 + "/plot_with_matplotlib"
            send_log(temp_key)
            fig = source_df.plot(kind="bar")
            st.pyplot()
            setup_screenshot_button(temp_key, "image")
            st.session_state.matplotlib = True

        if (
            st.checkbox("Box plot")
            #  or "boxplot" in st.session_state
        ):
            temp_key = depth1 + "/box_plot"
            send_log(temp_key)
            source_df.plot(
                kind="box",
                subplots=True,
                layout=(1, source_df.shape[1]),
                sharex=False,
                sharey=False,
                figsize=(source_df.shape[1] * 2 + 5, 5),
            )
            st.pyplot()
            setup_screenshot_button(temp_key, "image")
            st.session_state.boxplot = True

        # Show Plots
        if (
            st.checkbox("Correlation Plot with Seaborn ")
            # or "seaborn" in st.session_state
        ):
            temp_key = depth1 + "/correlation_plot_with_seaborn"
            send_log(temp_key)
            st.write(sns.heatmap(source_df.corr(), annot=True))
            # Use Matplotlib to render seaborn
            st.pyplot()
            setup_screenshot_button(temp_key, "image")
            st.session_state.seaborn = True

        # Show Plots
        if st.checkbox("Bar Plot of Groups") or "barplotgroups" in st.session_state:
            temp_key = depth1 + "/bar_plot_of_groups"
            v_counts = source_df.groupby(source_df.columns[-1]).mean()
            v_counts.plot(kind="bar")
            st.pyplot()
            send_log(temp_key)
            setup_screenshot_button(temp_key, "image")
            st.session_state.barplotgroups = True

        if st.checkbox("Bar Plot of Counts") or "barplotcounts" in st.session_state:
            temp_key = depth1 + "/bar_plot_of_counts"
            v_counts = source_df.groupby(source_df.columns[-1]).mean()
            st.bar_chart(v_counts)
            send_log(temp_key)
            setup_screenshot_button(temp_key, "vega_plot")
            st.session_state.barplotcounts = True


### --------------------------------------------------- Modelling Task ------------------------------------------- ###
if task == "Modeling":
    show_machine_learning_model(source_df)
    setup_screenshot_button(f"{dataset}/modeling", "dataframe")


### --------------------------------------------------- Prediction Task ------------------------------------------- ###
if task == "Prediction":
    send_log(f"{dataset}/prediction")
    st.sidebar.subheader("Prediction section")
    # pulished features/target
    features, labels = handle_io(source_df)
    df_test, clfs = user_input_features(features)
    st.subheader("User Input parameters")
    st.write(df_test)

    st.subheader("Prediction Probability")
    predictions = compute_predition(clfs, features, labels, df_test)

    # st.write(predictions)
    # predict = [dataset.target_names]
    # predict = [source_df.groupby(source_df.columns[-1]).size().index]
    # predict = [source_df.iloc[:, -1].unique()]
    predict = [labels.unique()]
    ind = ["Class label"]
    predict2 = []
    for i, p in predictions.items():
        predict.append(p[0].transpose())
        ind.append(i)
        predict2.append(p[1])
    df1 = pd.DataFrame(predict)
    df1.index = ind

    df2 = pd.DataFrame(predict2)
    if not df2.empty:
        df2.index = ind[1:]
        df2.columns = ["Output"]

    st.table(df1)
    st.table(df2)


### ---------------------------------------------- Image Manipulation ---------------------------------------------- ###
if task == "Image manipulation":
    log_action = f"{dataset}/image_manipulation"

    # Image Type
    species_type = st.radio(
        "What is the Species do you want to see?", source_df["variety"].unique()
    )

    if species_type == "Setosa":
        send_log(log_action + "/setosa")
        my_image = load_image("imgs/iris_setosa.jpg")
    elif species_type == "Versicolor":
        send_log(log_action + "/versicolor")
        my_image = load_image("imgs/iris_versicolor.jpg")
    elif species_type == "Virginica":
        send_log(log_action + "/virginica")
        my_image = load_image("imgs/iris_virginica.jpg")

    if st.sidebar.checkbox("Show original"):
        st.image(my_image)
        send_log(log_action + "/show_original")
        setup_screenshot_button(log_action + "/show_original", "image")

    if st.sidebar.checkbox("Change contrast"):
        enh = ImageEnhance.Contrast(my_image)
        img_width = st.sidebar.slider("Set Image Width", 300, 500)
        num = st.sidebar.slider("Set Your Contrast Number", 1.0, 3.0)
        st.image(enh.enhance(num), width=img_width)
        send_log(log_action + "/change_contrast")
        setup_screenshot_button(log_action + "/change_contrast", "image")


# print("Session state data: ", st.session_state)
