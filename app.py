import folium
import pickle
import streamlit as st

from streamlit_folium import st_folium

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