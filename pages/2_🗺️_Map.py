import csv
import folium
import streamlit as st
from streamlit_folium import st_folium

datafile = 'new_maps.csv'

@st.cache_data
def read_data():

    data=[]
    with open(datafile,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        print("reading...")
        for row in reader:
            latitude= row['Latitude']
            longitude = row['Longitude']
            data.append({
                'name':row['Name'],
                'contact_no':row['Contact Number'],
                'latitude':float(latitude),
                'longitude':float(longitude)
            })
        return data
data = read_data()

CONNECTICUT_CENTER = (19.16,72.84)
map = folium.Map(location=CONNECTICUT_CENTER, zoom_start=9)

for station in data:
    feature1 = station["name"],
    feature2 = station["contact_no"],
    popup_content = f"<h6>{feature1}<br>{feature2}</h6>"
    location = station['latitude'], station['longitude']
    folium.Marker(location, popup=popup_content).add_to(map)
st.header("Hospitals Near Me")

st_folium(map, width=1000)
# Sidebar with option to select a page
st.sidebar.success("Select a page above")

        