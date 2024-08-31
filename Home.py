import streamlit as st

st.set_page_config(
    page_title="Home - Football Stats",
    page_icon="⚽",
    layout="wide"
)

st.sidebar.title('Final Project')
st.title('Analizing Competitions and Offensive Stats')
st.write('Developed by Cristhian Camilo Beltrán Valencia')
st.write('Data from https://fbref.com/en/')
st.markdown('___________')

info = st.expander("**Info**", expanded=True)
with info:
    st.write('The app is designed to provide comprehensive information and insights on various football leagues across Latin America and Europe. It serves as a valuable tool for fans, analysts, and professionals by aggregating data on leagues, teams, and players. Users can access detailed statistics, performance trends, and historical records, allowing them to extract meaningful insights. The app also features comparative analysis tools, enabling users to explore and contrast different leagues, identify top-performing teams, and track player progress over time. With its user-friendly interface, the app offers a seamless experience for anyone looking to deepen their understanding of the football landscape in these regions. FBref Stats.')

info = st.expander("**Functionality**", expanded=True)
with info:
    st.write("The app operates in two main sections. The first section focuses on leagues and teams, pulling data from the FBref website using Python-based web scraping techniques. Here, users can explore performance metrics for each season, comparing teams across the selected leagues. Visualizations are generated in Python to make it easier to spot trends and gain insights. However, data availability may vary depending on what's accessible on the website, and not all leagues have consistent data across years. The second section is dedicated to player performance, offering similar functionality but focused on individual players. It allows users to track player stats across different leagues, with visual tools designed to highlight key performance indicators. As with teams, the data is scraped from FBref, and the availability may differ depending on the league and season.")
    