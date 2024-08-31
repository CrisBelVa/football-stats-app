import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# Load the data
file_path = '/Users/cristhianbeltran/Desktop/Sports_Data_Campus/Master/Proyecto_Final/Streamlit/data/player_stats_final.csv'
player_stats_final = pd.read_csv(file_path)

# Title of the app
st.title("Player Stats Comparison")

# Sidebar filters
st.sidebar.title("Filters")

# Position filter
positions = player_stats_final['Pos'].unique()
pos_selected = st.sidebar.multiselect("Select Position", positions, default=['FW'])

# Age range filter
age_min, age_max = player_stats_final['Age'].min(), player_stats_final['Age'].max()
age_selected = st.sidebar.slider('Select Age Range', min_value=int(age_min), max_value=int(age_max), value=(int(age_min), int(age_max)))

# Matches Played range filter
mp_min, mp_max = player_stats_final['MP'].min(), player_stats_final['MP'].max()
mp_selected = st.sidebar.slider('Select Matches Played Range', min_value=int(mp_min), max_value=int(mp_max), value=(int(mp_min), int(mp_max)))

# Minutes Played range filter
min_min, min_max = player_stats_final['Min'].min(), player_stats_final['Min'].max()
min_selected = st.sidebar.slider('Select Minutes Played Range', min_value=int(min_min), max_value=int(min_max), value=(int(min_min), int(min_max)))

# Apply filters to the DataFrame
df_filtered = player_stats_final[
    (player_stats_final['Age'] >= age_selected[0]) & (player_stats_final['Age'] <= age_selected[1]) &
    (player_stats_final['MP'] >= mp_selected[0]) & (player_stats_final['MP'] <= mp_selected[1]) &
    (player_stats_final['Min'] >= min_selected[0]) & (player_stats_final['Min'] <= min_selected[1]) &
    (player_stats_final['Pos'].isin(pos_selected))
]

# League selection
leagues = df_filtered['league'].unique()
selected_league = st.selectbox("Select League", leagues)

# Year selection based on selected league
years = df_filtered[df_filtered['league'] == selected_league]['Year'].unique()
selected_year = st.selectbox("Select Year", years)

# Team selection based on selected league and year
teams = df_filtered[(df_filtered['league'] == selected_league) & 
                    (df_filtered['Year'] == selected_year)]['Team'].unique()
selected_team = st.selectbox("Select Team", teams)

# Player selection based on selected team
players = df_filtered[(df_filtered['league'] == selected_league) & 
                      (df_filtered['Year'] == selected_year) & 
                      (df_filtered['Team'] == selected_team)]['Player'].unique()
selected_player = st.selectbox("Select Player", players)

# Display selected player's stats
player_data = df_filtered[(df_filtered['league'] == selected_league) & 
                          (df_filtered['Year'] == selected_year) & 
                          (df_filtered['Team'] == selected_team) & 
                          (df_filtered['Player'] == selected_player)]

st.write("Player Stats:", player_data)

# Create radar chart for the selected player's main metrics
metrics = ['Gls', 'Ast', 'xG', 'xAG', 'Gls/90', 'xG/90', 'PrgC', 'PrgP']
player_metrics = player_data[metrics].values.flatten()

# Normalize the metrics to make them comparable
def normalize_metrics(player_stats, metrics):
    normalized_stats = []
    for metric in metrics:
        max_val = player_stats_final[metric].max()
        min_val = player_stats_final[metric].min()
        value = player_stats[metric].values[0]
        normalized_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        normalized_stats.append(normalized_value)
    return normalized_stats

normalized_player_metrics = normalize_metrics(player_data, metrics)

# Radar chart function
def create_radar_chart(normalized_metrics, metrics, title=''):
    num_vars = len(metrics)
    
    # Compute angle of each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], metrics, color='grey', size=8)
    
    # Plot player metrics
    player_values = normalized_metrics + normalized_metrics[:1]
    ax.plot(angles, player_values, linewidth=1, linestyle='solid', label='Player 1')
    ax.fill(angles, player_values, 'b', alpha=0.1)

    # Title
    plt.title(title, size=15, color='blue', y=1.1)

    return fig

# Display radar chart for the selected player
st.subheader(f"{selected_player}'s Radar Chart")
radar_chart = create_radar_chart(normalized_player_metrics, metrics, title=selected_player)
st.pyplot(radar_chart)

# Option to compare with another player
st.subheader("Compare with Another Player")

# League selection for comparison
comp_leagues = df_filtered['league'].unique()
comp_selected_league = st.selectbox("Select League for Comparison", comp_leagues)

# Year selection based on selected league for comparison
comp_years = df_filtered[df_filtered['league'] == comp_selected_league]['Year'].unique()
comp_selected_year = st.selectbox("Select Year for Comparison", comp_years)

# Team selection based on selected league and year for comparison
comp_teams = df_filtered[(df_filtered['league'] == comp_selected_league) & 
                         (df_filtered['Year'] == comp_selected_year)]['Team'].unique()
comp_selected_team = st.selectbox("Select Team for Comparison", comp_teams)

# Player selection based on selected team for comparison
comp_players = df_filtered[(df_filtered['league'] == comp_selected_league) & 
                           (df_filtered['Year'] == comp_selected_year) & 
                           (df_filtered['Team'] == comp_selected_team)]['Player'].unique()
comp_selected_player = st.selectbox("Select Player for Comparison", comp_players)

# Display selected comparison player's stats
comp_player_data = df_filtered[(df_filtered['league'] == comp_selected_league) & 
                               (df_filtered['Year'] == comp_selected_year) & 
                               (df_filtered['Team'] == comp_selected_team) & 
                               (df_filtered['Player'] == comp_selected_player)]

st.write("Comparison Player Stats:", comp_player_data)

# Create radar chart for the comparison player's main metrics
comp_player_metrics = comp_player_data[metrics].values.flatten()
normalized_comp_player_metrics = normalize_metrics(comp_player_data, metrics)

# Update radar chart with comparison player
def update_radar_chart(fig, normalized_comp_metrics, metrics, label=''):
    num_vars = len(metrics)
    
    # Compute angle of each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Plot comparison player metrics
    comp_values = normalized_comp_metrics + normalized_comp_metrics[:1]
    ax = fig.gca()
    ax.plot(angles, comp_values, linewidth=1, linestyle='solid', label=label)
    ax.fill(angles, comp_values, 'r', alpha=0.1)
    
    # Add legend
    ax.legend(loc='upper left')

    return fig

st.subheader(f"Comparison: {selected_player} vs {comp_selected_player}")
comp_radar_chart = update_radar_chart(radar_chart, normalized_comp_player_metrics, metrics, label=comp_selected_player)
st.pyplot(comp_radar_chart)
