# imports 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from pathlib import Path
import sklearn as skl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

st.set_option('deprecation.showfileUploaderEncoding', False)

# page layout
st.set_page_config(page_title="Machine Learning App", page_icon=":tada:", layout="wide")

st.subheader("THe Machine Learning App")
st.title("Measure and Prediction in the Arsenic Contaimination in Water ")
st.write("Implement LogisticRegression() function to predict the level of arsenic")

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['LAT', 'LON', 'WELL_DEPTH', 'UNDER_5', 'CURRENT_STANDARD', '1942_STANDARD'])
st.map()


#side bars
with st.sidebar.header('Upload your CSV data'):uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# build machine model 
def build_model(df):
    X = df[['LAT', 'LON']]
    X = X.to_numpy()
    y = df[['UNDER_5']].values.reshape(-1, 1)
    y_cur = df[['CURRENT_STANDARD']].values.reshape(-1, 1)
    y_1942 = df[['1942_STANDARD']].values.reshape(-1, 1)

    with open('under5.pkl', 'rb') as f:
        under5 = pickle.load(f)

    with open('current.pkl', 'rb') as f:
        current = pickle.load(f)

    with open('oldStandard.pkl', 'rb') as f:
        oldStandard = pickle.load(f)

    st.write('Accuracy score for under 5 ㎍/L')
    st.info(under5.predict(X[0:1]))

    st.write('Accuracy score for 10 ㎍/L')
    st.info(current.predict(X[0:1]))

    st.write('Accuracy score for 50 ㎍/L')
    st.info(oldStandard.predict(X[0:1]))


# display page 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')