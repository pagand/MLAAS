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
from PIL import Image,ImageFilter,ImageEnhance

import pathlib
import logging
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import streamlit.components.v1 as components
from pandas_profiling import ProfileReport

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()



### --------------------------------------------------- Functions def ---------------------------------------------- ###
# Image Manipulation
@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
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

def user_input_features(features):
    data = {}
    for _, j in enumerate(features.columns):
        var0 = st.sidebar.slider(j,float(features[j].min()), float(features[j].max()), float(features[j].mean()))
        # print(float(features[j].mean()))
        data[j] = var0

    features = pd.DataFrame(data, index=[0])
    clfs = []
    clfs = st.multiselect('Select models', ('Random Forest', 'Logistic Regression', "Support Vector", "Naive Bayes",
                                            "Decision Tree", "K Nearest Neighbour", "linear discriminant"))
    return features, clfs

@st.cache(persist=True)
def compute_predition(clfs, features, labels, df_test):
    models = []
    predictions = {}
    if 'Random Forest' in clfs:
        models.append(('RF',RandomForestClassifier()))
    if 'Logistic Regression' in clfs:
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    if 'Support Vector' in clfs:
        models.append(('SVM', SVC(gamma='auto', probability=True)))
    if 'Naive Bayes' in clfs:
        models.append(('NB', GaussianNB()))
    if 'Decision Tree' in clfs:
        models.append(('CART', DecisionTreeClassifier()))
    if 'K Nearest Neighbour' in clfs:
        models.append(('KNN', KNeighborsClassifier()))
    if 'linear discriminant' in clfs:
        models.append(('LDA', LinearDiscriminantAnalysis()))

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

    fig = px.scatter(selected_species_df, x=feature_x, y=feature_y, color=selected_species_df.columns[-1])
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
    fig2 = px.histogram(selected_species_df, x=feature, color=selected_species_df.columns[-1], marginal="rug")
    st.plotly_chart(fig2)


def _handle_missing(features, labels):
    appraoch = st.sidebar.radio('Handle missing data',('impute with mean','Replace with zero', 'drop the entery'))
    if appraoch == 'impute with mean':
        # approach 1 (imputing with the mean)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        new_features = features.fillna(features.mean())
        new_labels = labels.fillna(labels.mean())
    elif appraoch == 'drop the entery':
        # approach 2 (delete the row of the nan values)
        #TODO consider changing the output size as well
        dum = pd.concat([features, labels], axis=1, join='inner').dropna(axis=0)
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
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            st.warning("Colomn <{}> is not numeric. We have mapped it to numeric values.".format(col))
            n = features.groupby(col).ngroups
            Map = {key: index/(n-1) for index, key in enumerate(features.groupby(col).groups.keys())}
            features[col] = features[col].map(Map, na_action=None).astype(float)


    labels = source_df.iloc[:, -1]
    # check if all the colomn are numeric
    # if not pd.api.types.is_numeric_dtype(labels):
    #     print('bye')
    #     st.warning("Output value is not numeric. We have mapped it to numeric values.")
    #     Map = {key: index/(labels.groupby(level=0).ngroups-1) for index, key in enumerate(labels.groupby(level=0).groups.keys())}
    #     labels = labels.map(Map, na_action=None).astype(float)

    if labels.isnull().values.any() or features.isnull().values.any():
        st.warning("There are null values in your dataset. Please choose a method in the side bar to handle it.")
        features, labels = _handle_missing(features, labels)

    return  features, labels


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

    ratio = st.sidebar.slider('Train-test ratio', 0.0, 1.0, 0.7)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=ratio, random_state=1
    )

    alg = ["Random Forest", "Logistic Regression", "Decision Tree", "Support Vector Machine", "Naive Bayes",
           "K Nearest Neighbour" , "linear discriminant"]
    classifier = st.selectbox("Which algorithm?", alg)

    if classifier == "Random Forest":
        model = RandomForestClassifier()
    elif classifier == "Logistic Regression":
        model =  LogisticRegression(solver='liblinear', multi_class='ovr')
    elif classifier == "Support Vector Machine":
        model = SVC(gamma='auto', probability=True)
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
st.sidebar.info("Please choose your dataset and the task.")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.header('User input')
dataset = st.sidebar.radio('Please select your dataset',('Iris','PIMA', 'Wine', 'External'))
### --------------------------------------------------- Dataset ---------------------------------------------------- ###
DATA_URL = ""
if dataset == 'Iris':
    DATA_URL = "https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/gallery/iris_classification/iris.csv"
elif dataset == 'PIMA':
    DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
elif dataset == 'Wine':
    DATA_URL = "https://raw.githubusercontent.com/reubengazer/Wine-Quality-Analysis/master/winequalityN.csv"
else:
    type = st.sidebar.radio('Please select', ('Upload', 'URL'))
    if type == 'Upload':
        DATA_URL = st.sidebar.file_uploader("Choose CSV file", type=['csv'], accept_multiple_files=False)
    else:
        DATA_URL = st.sidebar.text_input("Input remote URL (raw format)")
if DATA_URL:
    try:
        source_df =  read_csv(DATA_URL)
    except:
        st.error("The file/URL is not parsable.")
        st.warning("Please select another file/URL or choose the preloaded dataset.")
        DATA_URL = ""

### --------------------------------------------------- Task  ------------------------------------------------------ ###
task = ''
if dataset == 'Iris':
    task = st.sidebar.radio('Please select your task',('Data review','Visualization', 'Modeling','Prediction', 'Image manipulation'))
elif DATA_URL:
    task = st.sidebar.radio('Please select your task',
                            ('Data review', 'Visualization', 'Modeling', 'Prediction'))

st.sidebar.write("""
---
""")



### --------------------------------------------------- Data review  ----------------------------------------------- ###
if task == 'Data review':
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
    data_dim = st.radio('What Dimension Do You Want to Show', ('Rows', 'Columns'))
    if data_dim == 'Rows':
        st.text("Number of data points")
        st.write(len(source_df))
    if data_dim == 'Columns':
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
if task == 'Visualization':
    selected_species_df = select_species(source_df)
    if not selected_species_df.empty:
        show_scatter_plot(selected_species_df)
        show_histogram_plot(selected_species_df)

    if st.sidebar.checkbox("Group plots"):
        # Show Plots
        if st.checkbox("Plot with Matplotlib (For large dataset, this might take a while) "):
            fig = source_df.plot(kind='bar')
            st.pyplot()

        if st.checkbox("Box plot"):
            source_df.plot(kind='box', subplots=True, layout=(1,source_df.shape[1]), sharex=False, sharey=False,  figsize=(source_df.shape[1]*2+5, 5))
            st.pyplot()

        # Show Plots
        if st.checkbox("Correlation Plot with Seaborn "):
            logging.info('Plot with Matplotlib')
            st.write(sns.heatmap(source_df.corr(),annot=True))
            # Use Matplotlib to render seaborn
            st.pyplot()

        # Show Plots
        if st.checkbox("Bar Plot of Groups"):
            v_counts = source_df.groupby(source_df.columns[-1]).mean()
            v_counts.plot(kind='bar')
            st.pyplot()

        if st.checkbox("Bar Plot of  Counts"):
            v_counts = source_df.groupby(source_df.columns[-1]).mean()
            # print(v_counts)
            st.bar_chart(v_counts)



### --------------------------------------------------- Modeling Task ------------------------------------------- ###
if task =='Modeling':
    show_machine_learning_model(source_df)


### --------------------------------------------------- Prediction Task ------------------------------------------- ###
if task =='Prediction':
    st.sidebar.subheader('Prediction section')
    # pulished features/target
    features, labels = handle_io(source_df)
    df_test, clfs = user_input_features(features)
    st.subheader('User Input parameters')
    st.write(df_test)



    st.subheader('Prediction Probability')
    predictions = compute_predition(clfs, features, labels, df_test)

    #st.write(predictions)
    #predict = [dataset.target_names]
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

### --------------------------------------------------- Image Manipulation ----------------------------------------- ###
if task =='Image manipulation':
    # Image Type
    species_type = st.radio('What is the Species do you want to see?', source_df["variety"].unique())

    if species_type == 'Setosa':
        my_image = load_image('imgs/iris_setosa.jpg')

    elif species_type == 'Versicolor':
        my_image = load_image('imgs/iris_versicolor.jpg')
    elif species_type == 'Virginica':
        my_image = load_image('imgs/iris_virginica.jpg')

    if st.sidebar.checkbox("Show original"):
        st.image(my_image)


    if st.sidebar.checkbox("Change contrast"):
        enh = ImageEnhance.Contrast(my_image)
        img_width = st.sidebar.slider("Set Image Width",300,500)
        num = st.sidebar.slider("Set Your Contrast Number", 1.0, 3.0)
        st.image(enh.enhance(num),width=img_width)