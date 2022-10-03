import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance

import pathlib
import logging
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

<<<<<<< HEAD
import streamlit.components.v1 as components
from pandas_profiling import ProfileReport

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
=======
logging.basicConfig(
    filename="app.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
>>>>>>> 302002b7e16e392ba2c90df265b214b90cef1e4d
logger = logging.getLogger()


### --------------------------------------------------- Functions def ---------------------------------------------- ###
# Image Manipulation
@st.cache
def load_image(img):
    im = Image.open(os.path.join(img))
    return im


@st.cache
def show_html(HtmlFile):
    return HtmlFile.read() 

@st.cache
def read_csv(DATA_URL) -> pd.DataFrame:
    """## Iris DataFrame

    Returns:
        pd.DataFrame -- A dataframe with the source iris data
    """
    return pd.read_csv(DATA_URL)


def user_input_features():
    # if dataset == "Iris":
    #     sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    #     sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    #     petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    #     petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    #     data = {'sepal_length': sepal_length,
    #             'sepal_width': sepal_width,
    #             'petal_length': petal_length,
    #             'petal_width': petal_width}
    data = {}
    for i, j in enumerate(source_df.columns[0:-1]):
        var0 = st.sidebar.slider(
            j,
            float(source_df[j].min()),
            float(source_df[j].max()),
            float(source_df[j].mean()),
        )
        print(float(source_df[j].mean()))
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


@st.cache(persist=True)
def compute_predition(clfs, source_df, df_test):
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

    # features = source_df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
    features = source_df.iloc[:, :-1]
    # labels = source_df["variety"].values
    labels = source_df.iloc[:, -1]

    # Train on all data
    X = features
    Y = labels
    # X, x_test, Y, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

    for name, model in models:
        model.fit(X, Y)
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


def show_machine_learning_model(source_df: pd.DataFrame):
    """Component to show the performance of an ML Algo trained on the iris data set

    Arguments:
        source_df {pd.DataFrame} -- The source iris data set

    Raises:
        NotImplementedError: Raised if a not supported model is selected
    """
    st.header("Machine Learning models")
    # features = source_df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
    # labels = source_df["variety"].values
    features = source_df.iloc[:, :-1]
    labels = source_df.iloc[:, -1]
    ratio = st.sidebar.slider("Train-test ratio", 0.0, 1.0, 0.7)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=ratio, random_state=1
    )
    alg = [
        "Random Forest",
        "Logistic Regression",
        "Decision Tree",
        "Support Vector Machine",
        "Naive Bayes",
        "K Nearest Neighbour",
        "linear discriminant",
    ]
    classifier = st.selectbox("Which algorithm?", alg)

    if classifier == "Random Forest":
        model = RandomForestClassifier()
    elif classifier == "Logistic Regression":
        model = LogisticRegression(solver="liblinear", multi_class="ovr")
    elif classifier == "Support Vector Machine":
        model = SVC(gamma="auto", probability=True)
    elif classifier == "Naive Bayes":
        model = GaussianNB()
    elif classifier == "Decision Tree":
        model = DecisionTreeClassifier()
    elif classifier == "K Nearest Neighbour":
        model = KNeighborsClassifier()
    elif classifier == "linear discriminant":
        model = LinearDiscriminantAnalysis()
    else:
        raise NotImplementedError()

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    st.write("Accuracy: ", acc.round(2))
    pred_model = model.predict(x_test)
    cm_model = confusion_matrix(y_test, pred_model)
    st.write("Confusion matrix: ", cm_model)


### --------------------------------------------------- MAIN ------------------------------------------------------- ###
st.title("DaTu EDA/MLAAS toolbox")
st.info("Please choose your dataset and the task from the left bar menu.")

st.set_option("deprecation.showPyplotGlobalUse", False)

st.sidebar.header("User input")
dataset = st.sidebar.radio("Please select your dataset", ("Iris", "PIMA", "External"))
### --------------------------------------------------- Dataset ---------------------------------------------------- ###
DATA_URL = ""
if dataset == "Iris":
    DATA_URL = "https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/gallery/iris_classification/iris.csv"
<<<<<<< HEAD
elif dataset == 'PIMA':
    DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
=======
elif dataset == "PIMA":
    DATA_URL = (
        "https://gist.githubusercontent.com/chaityacshah/899a95deaf8b1930003ae93944fd17d7"
        "/raw/3d35de839da708595a444187e9f13237b51a2cbe/pima-indians-diabetes.csv"
    )
>>>>>>> 302002b7e16e392ba2c90df265b214b90cef1e4d
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
if dataset == "Iris":
    task = st.sidebar.radio(
        "Please select your task",
        (
            "Data review",
            "Visualization",
            "Modeling",
            "Prediction",
            "Image manipulation",
        ),
    )
elif DATA_URL:
    task = st.sidebar.radio(
        "Please select your task",
        ("Data review", "Visualization", "Modeling", "Prediction"),
    )

st.sidebar.write(
    """
---
"""
)


### --------------------------------------------------- Data review  ----------------------------------------------- ###
if task == "Data review":
    # Show Dataset
    if st.checkbox("Preview DataFrame"):
        if st.button("Head"):
            st.write(source_df.head())
        if st.button("Tail"):
            st.write(source_df.tail())

    # Show Entire Dataframe
    if st.checkbox("Show All DataFrame"):
        st.dataframe(source_df)

    # Dimensions
    data_dim = st.radio("What Dimension Do You Want to Show", ("Rows", "Columns"))
    if data_dim == "Rows":
        st.text("Number of data points")
        st.write(len(source_df))
    if data_dim == "Columns":
        st.text("Number of features")
        st.write(source_df.shape[1])
        st.write(source_df.columns)

    if st.checkbox("Show Summary of Dataset"):
        st.write(source_df.describe())

    if st.checkbox("Categories"):
        st.write(source_df.groupby(source_df.columns[-1]).size())


    if st.checkbox("Report (This might take a while for heavy datasets)"):
        if dataset == "External":
            file = ProfileReport(source_df)
            file.to_file(output_file="output_{}.html".format(dataset))
        try: 
            with open("output_{}.html".format(dataset), 'r', encoding='utf-8') as HtmlFile:
                source_code = show_html(HtmlFile)
        except:
            file = ProfileReport(source_df)
            file.to_file(output_file="output_{}.html".format(dataset))
            with open("output_{}.html".format(dataset), 'r', encoding='utf-8') as HtmlFile:
                source_code = show_html(HtmlFile)
        components.html(source_code, height = 600, scrolling=True)
        
                




### --------------------------------------------------- Visualization  --------------------------------------------- ###
if task == "Visualization":
    selected_species_df = select_species(source_df)
    if not selected_species_df.empty:
        show_scatter_plot(selected_species_df)
        show_histogram_plot(selected_species_df)

    if st.sidebar.checkbox("Group plots"):
        # Show Plots
<<<<<<< HEAD
        if st.checkbox("Plot with Matplotlib (For large dataset, this might take a while) "):
            fig = source_df.plot(kind='bar')
            st.pyplot()

        if st.checkbox("Box plot"):
            source_df.plot(kind='box', subplots=True, layout=(1,source_df.shape[1]), sharex=False, sharey=False,  figsize=(source_df.shape[1]*2+5, 5))
            st.pyplot()

=======
        if st.checkbox("Plot with Matplotlib "):
            fig = source_df.plot(kind="bar")
            st.pyplot()

>>>>>>> 302002b7e16e392ba2c90df265b214b90cef1e4d
        # Show Plots
        if st.checkbox("Correlation Plot with Seaborn "):
            logging.info("Plot with Matplotlib")
            st.write(sns.heatmap(source_df.corr(), annot=True))
            # Use Matplotlib to render seaborn
            st.pyplot()

        # Show Plots
        if st.checkbox("Bar Plot of Groups"):
            v_counts = source_df.groupby(source_df.columns[-1]).mean()
            v_counts.plot(kind="bar")
            st.pyplot()

        if st.checkbox("Bar Plot of  Counts"):
            v_counts = source_df.groupby(source_df.columns[-1]).mean()
            # print(v_counts)
            st.bar_chart(v_counts)


### --------------------------------------------------- Prediction Task ------------------------------------------- ###
if task == "Modeling":
    show_machine_learning_model(source_df)


### --------------------------------------------------- Prediction Task ------------------------------------------- ###
if task == "Prediction":
    st.sidebar.subheader("Input features for prediction")
    df_test, clfs = user_input_features()
    st.subheader("User Input parameters")
    st.write(df_test)

    st.subheader("Prediction Probability")
    predictions = compute_predition(clfs, source_df, df_test)

    # st.write(predictions)
    # predict = [dataset.target_names]
    # predict = [source_df.groupby(source_df.columns[-1]).size().index]
    predict = [source_df.iloc[:, -1].unique()]
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

### --------------------------------------------------- Image Manipulation ----------------------------------------- ###
if task == "Image manipulation":
    # Image Type
    species_type = st.radio(
        "What is the Species do you want to see?", source_df["variety"].unique()
    )

    if species_type == "Setosa":
        my_image = load_image("imgs/iris_setosa.jpg")

    elif species_type == "Versicolor":
        my_image = load_image("imgs/iris_versicolor.jpg")
    elif species_type == "Virginica":
        my_image = load_image("imgs/iris_virginica.jpg")

    if st.sidebar.checkbox("Show original"):
        st.image(my_image)

    if st.sidebar.checkbox("Change contrast"):
        enh = ImageEnhance.Contrast(my_image)
        img_width = st.sidebar.slider("Set Image Width", 300, 500)
        num = st.sidebar.slider("Set Your Contrast Number", 1.0, 3.0)
        st.image(enh.enhance(num), width=img_width)
