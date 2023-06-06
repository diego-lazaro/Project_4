## imports 
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
from PIL import Image
import folium

from streamlit_folium import st_folium

## initate 
pic1 = Image.open("arsenic_chem.jpg")
pic2 = Image.open("arsenic.jpg")
pic3 = Image.open("arsenic_posion.png")



## page general layout
st.set_page_config(page_title="Machine Learning App", page_icon=":tada:", layout="wide")

st.subheader("The Machine Learning App")
st.title("Analyzed arsenic contamination levels of ground water within the United States ")
st.write("Implement LogisticRegression() function to train 3 different models of spectrum of contamination. The three spectrums created were, a 1942 Standard in United States of 50㎍/L, a current United States aresenic standard of less than 10㎍/L, and concentrations less than 5㎍/L.")



## project main page displays the following

#if no file upload trains the model using pickle
user_lat = st.number_input('Latitude', value=38.54, min_value=24.5, max_value=50.0, step=(1/3600))
st.write('Latitude: ', user_lat, '°')

user_lon = st.number_input('Longitude', value=-121.75, min_value=-125.0, max_value=-67.0, step=(1/3600))
st.write('Longitude: ', user_lon, '°')

# center & add marker
m = folium.Map(location=[user_lat, user_lon], zoom_start=16)
folium.Marker(
    [user_lat, user_lon], popup="You are Here", tooltip="You are Here"
).add_to(m)

# call to render Folium map in Streamlit
st_data = st_folium(m, width=725)

with open('Resources/Trained_Models/oldStandard.pkl', 'rb') as f:
    oldStandard = pickle.load(f)

with open('Resources/Trained_Models/current.pkl', 'rb') as f:
    current = pickle.load(f)

with open('Resources/Trained_Models/under5.pkl', 'rb') as f:
    under5 = pickle.load(f)

over50 = oldStandard.predict([[user_lat, user_lon]])
over10 = current.predict([[user_lat, user_lon]])
over5 = under5.predict([[user_lat, user_lon]])

with st.container():
    if (over50 == 1):
        st.write('<h3>The Arsenic Contamination of the Groundwater at this location is very high!</h3><center><h1>(over 50㎍/L)</h1></center></br><h5>This exceeds the Environmental Protection Agency\'s orignal arsenic standard set in 1942!!</h5>', unsafe_allow_html=True)
    else:
        if (over10 == 1):
            st.write('<h3>The Arsenic Contamination of the Groundwater at this location is high!</h3><center><h3>(over 10㎍/L but under 50㎍/L)</h3></center></br><h5>This exceeds the Environmental Protection Agency\'s arsenic standard as of January 2006</h5>', unsafe_allow_html=True)
        else:
            if (over5 == 1):
                st.write('<h3>The Arsenic Contamination of the Groundwater at this location is . . .</h3><center><h1>(between 5㎍/L and 10㎍/L)</h1></center></br><h5>This meets the Environmental Protection Agency\'s arsenic standard as of January 2006</h5>', unsafe_allow_html=True)
            else:
                st.write('<h3>The Arsenic Contamination of the Groundwater at this location is . . .</h3><center><h1>(less than 5㎍/L)</h1></center></br><h5>This meets the Environmental Protection Agency\'s arsenic standard as of January 2006</h5>', unsafe_allow_html=True)

# displaying hyperlinks 
st.markdown("***")
st.write('<h1>Additional Resources </h1>',unsafe_allow_html=True)
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(pic1)
    
    with text_column:
        st.subheader("Arsenic Element Information")
        st.write("Arsenic definition, symbol, use, and facts")
        st.markdown("[Arsenic information . . .](https://www.britannica.com/science/arsenic)")

with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(pic2)
    
    with text_column:
        st.subheader("Arsenic Extra Element Information")
        st.write("Arsenic is a semi-metal. In its metallic form it is bright, silver-grey and brittle. Arsenic is a well-known poison. Arsenic compounds are sometimes used as rat poisons and insecticides but their use is strictly controlled.")
        st.markdown("[Warning Watch Video . . .](https://www.youtube.com/watch?v=BVht6JtufYE)")

with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(pic3)
    
    with text_column:
        st.subheader("Arsenic Poisoning Cases")
        st.write("EPA Drinking Water Standard for Arsenic ")
        st.markdown("[Read case here . . .](https://nepis.epa.gov/Exe/ZyPdf.cgi?Dockey=20001XXC.txt)")