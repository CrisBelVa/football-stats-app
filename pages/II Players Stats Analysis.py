import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
SEED = 1234
sns.set_theme(style="whitegrid")

file_path = '/Users/cristhianbeltran/Desktop/Sports Data Campus/Master Big Data aplicado al Scouting en Futbol/Proyecto Final/Streamlit/data/player_stats_final.csv'

df = pd.read_csv(file_path)

player_stats_final = df

st.title("Player Stats")
st.write('Squad Standard Stats FBref - Select on the Side Bar')
st.markdown("__________")

with st.sidebar.title("Select League, Year, Position, Age, Matches, Played, Minutes Played"):

    comp = df['league'].unique().tolist()
    comp_selected = st.sidebar.multiselect('League', options = comp, default=comp)

    years = df['Year'].unique().tolist()
    years_selected = st.sidebar.multiselect('Year', options=years, default=[2023])
    
    pos = df['Pos'].unique().tolist()
    pos_selected = st.sidebar.multiselect('Position', options = pos, default=pos)

    age_min, age_max = df['Age'].min(), df['Age'].max()
    mp_min, mp_max = df['MP'].min(), df['MP'].max()
    min_min, min_max = df['Min'].min(), df['Min'].max()
    
    age_selected = st.sidebar.slider('Select Age Range', min_value=age_min, max_value=age_max, value=(age_min, age_max))

    mp_selected = st.sidebar.slider('Select Matches Played Range', min_value=mp_min, max_value=mp_max, value=(mp_min, mp_max))
    
    min_selected = st.sidebar.slider('Select Minutes Played Range', min_value=min_min, max_value=min_max, value=(min_min, min_max))

df_filtered = df[
(df['Age'] >= age_selected[0]) & (df['Age'] <= age_selected[1]) &
(df['MP'] >= mp_selected[0]) & (df['MP'] <= mp_selected[1]) &
(df['Min'] >= min_selected[0]) & (df['Min'] <= min_selected[1]) &
(df['Pos'].isin(pos_selected)) &
(df['league'].isin(comp_selected)) &
(df['Year'].isin(years_selected))  # Filter by selected positions
]
st.dataframe(df_filtered)

#Navigation Menu
st.markdown("### **Select Analysis Category**")
category = st.selectbox("", ["Select a Category", "Goals and xG", "Assists", "Progression", "Clusters"])

if category == "Select a Category":
    st.warning("Please select a category from the dropdown above to view the analysis.")

# Goals and xG Analysis
elif category == "Goals and xG":
    st.subheader("Goals and xG Analysis")
# Goals and xG
    with st.expander("View Goals and xG by Player"):
        league_palette = {
    "liga_argentina": "blue",
    "brasileirao": "green",
    "eredivisie": "orange",
    "liga_mx": "red",
    "primeira_liga": "purple"
}
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'Gls': 'sum', 
        'xG': 'sum'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Gls', y='xG', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Goals (Gls)')
        ax.set_ylabel('Expected Goals (xG)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['Gls'].iloc[i] + 0.05, 
                y=player_stats['xG'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)


# Goal and xG Per Match
    with st.expander("View Goals and xG per Match by Player"):
  
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'Gls/90': 'mean', 
        'xG/90': 'mean'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Gls/90', y='xG/90', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Goals Per 90 minutes (Gls/90)')
        ax.set_ylabel('Expected Goals Per 90 minutes  (xG/90)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['Gls/90'].iloc[i] + 0.05, 
                y=player_stats['xG/90'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Goals and non penalty Goals
    with st.expander("View Players with with more than 15 Goals and Non-Penalty Goals"):
  
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
            'Gls': 'sum', 
            'G-PK': 'sum'
        }).reset_index()
    
        # Calculate penalty goals for clarity in stacking
        player_stats['PK_Gls'] = player_stats['Gls'] - player_stats['G-PK']
    
        # Filter to show only players with more than 15 goals
        player_stats = player_stats[player_stats['Gls'] > 15]
    
        # Sort data to improve visualization (optional)
        player_stats = player_stats.sort_values(by='Gls', ascending=False)
    
        # Create a horizontal bar plot for total goals (Gls)
        fig, ax = plt.subplots(figsize=(10, 12))
        sns.set_theme(style="whitegrid")
        
        # Plot for non-penalty goals (G-PK)
        sns.barplot(
            x='G-PK', y='Player', 
            color='blue', 
            label='Non-Penalty Goals',
            data=player_stats, 
            ax=ax
        )
        
        # Overlay the penalty goals on top of non-penalty goals
        for index, row in player_stats.iterrows():
            ax.barh(row['Player'], row['PK_Gls'], left=row['G-PK'], color='red', label='Penalty Goals' if index == 0 else "")
    
        # Add annotations for each bar
        for i in range(len(player_stats)):
            ax.text(player_stats['Gls'].iloc[i] + 0.5, i, f"{player_stats['Gls'].iloc[i]}", color='black', ha="center")
            ax.text(player_stats['G-PK'].iloc[i] + 0.5, i, f"{player_stats['G-PK'].iloc[i]}", color='white', ha="center")
    
        plt.title("Total Goals and Non-Penalty Goals per Player (More than 15 Goals)")
        plt.xlabel("Goals")
        plt.ylabel("Player")
        plt.legend(loc='lower right')
        st.pyplot(fig)


# Assists Analysis
elif category == "Assists":
    st.subheader("Assists Analysis")

#Assists vs xAG

    with st.expander("View Assists and xAG by Player"):
        league_palette = {
    "liga_argentina": "blue",
    "brasileirao": "green",
    "eredivisie": "orange",
    "liga_mx": "red",
    "primeira_liga": "purple"
}
  
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'Ast': 'sum', 
        'xAG': 'sum'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Ast', y='xAG', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Assists (Ast)')
        ax.set_ylabel('Expected Assists Goals (xAG)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['Ast'].iloc[i] + 0.05, 
                y=player_stats['xAG'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)


#Assists vs xAG Per Match

    with st.expander("View Assists and xAG Per Match by Player"):
  
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'Ast/90': 'mean', 
        'xAG/90': 'mean'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Ast/90', y='xAG/90', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Assists Per 90 Minutes(Ast/90)')
        ax.set_ylabel('Expected Assists Goals Per 90 Minutes (xAG/90)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['Ast/90'].iloc[i] + 0.05, 
                y=player_stats['xAG/90'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Goals and Assists
    with st.expander("View Goals and Assists by Player"):
  
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'Gls': 'sum', 
        'Ast': 'sum'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Gls', y='Ast', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Goals (Gls)')
        ax.set_ylabel('Assists (Ast)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['Gls'].iloc[i] + 0.05, 
                y=player_stats['Ast'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Progression Analysis
elif category == "Progression":
    st.subheader("Progression Analysis")

    #Progressive Carries and Progressive Passes
    with st.expander("View Progressive Carries and Progressive Passes"):
        league_palette = {
    "liga_argentina": "blue",
    "brasileirao": "green",
    "eredivisie": "orange",
    "liga_mx": "red",
    "primeira_liga": "purple"
}
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'PrgC': 'sum', 
        'PrgP': 'sum'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='PrgP', y='PrgC', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Progressive Carries (PrgP)')
        ax.set_ylabel('Progressive Passes (PrgC)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['PrgP'].iloc[i] + 0.05, 
                y=player_stats['PrgC'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        #Progressive Passes and Progressive Passes Received
    with st.expander("View Progressive Passes and Progressive Passes Received"):
        league_palette = {
    "liga_argentina": "blue",
    "brasileirao": "green",
    "eredivisie": "orange",
    "liga_mx": "red",
    "primeira_liga": "purple"
}
        player_stats = df_filtered.groupby(['Player', 'Team', 'league']).agg({
        'PrgP': 'sum', 
        'PrgR': 'sum'
        }).reset_index()

            # Create a scatter plot of xG vs. Goals for the current year
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='PrgP', y='PrgR', 
            hue='league',  # Use league for color coding
            palette=league_palette,  # Apply custom colors
            style='league',  # Use different markers for different leagues (optional)
            data=player_stats, 
            s=100,
            ax=ax
        )
        
        ax.set_xlabel('Progressive Passes (PrgP)')
        ax.set_ylabel('Progressive Passes Received (PrgR)')
        ax.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
    
        # Annotate each point with the player name and team
        for i in range(player_stats.shape[0]):
            ax.text(
                x=player_stats['PrgP'].iloc[i] + 0.05, 
                y=player_stats['PrgR'].iloc[i], 
                s=f"{player_stats['Player'].iloc[i]} ({player_stats['Team'].iloc[i]})",
                fontsize=9
            )
    
        # Customize legend
        ax.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Clusters Analysis
elif category == "Clusters":
    st.subheader("Clusters Analysis - Players")

    # Apply the filter using the sidebar selections
    df_filtered = df[
        (df['Age'] >= age_selected[0]) & (df['Age'] <= age_selected[1]) &
        (df['MP'] >= mp_selected[0]) & (df['MP'] <= mp_selected[1]) &
        (df['Min'] >= min_selected[0]) & (df['Pos'].isin(pos_selected)) &
        (df['league'].isin(comp_selected)) & (df['Year'].isin(years_selected))
    ]

    # Check if there's data to process after filtering
    if df_filtered.empty:
        st.warning("No data available for the selected leagues and years.")
    else:
# PCA
        with st.expander("PCA - Graphical representation of teams from the 5 analyzed leagues using 2 CPs"):
            
            off_metrics = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls/90', 'Ast/90', 'G+A/90', 'xG/90', 'xAG/90', 'xG+xAG']
            data_selection = df_filtered[['Player', 'Team', 'Year'] + off_metrics]  # Use filtered data
            data_preparation = data_selection.copy()
            data_preparation.set_index(['Player', 'Team', 'Year'], inplace=True)
            

            scaler = StandardScaler()
            data_preparation_scaled = scaler.fit_transform(data_preparation)

            # PCA process
            nOPT = 4
            pca = PCA(n_components=nOPT)
            new_scores = pca.fit_transform(data_preparation_scaled)
            data_final = pd.DataFrame(new_scores, index=data_preparation.index)  # Use the index from data_preparation
            data_final.columns = ["CP_" + str(i) for i in range(1, nOPT + 1)]

            # Prepare the final plot data
            data_final_plot = data_final.reset_index()  # Reset index to get 'Player' and 'Year' as columns

            # Streamlit setup
            st.subheader("PCA Analysis - 2 CPs")

            # PCA plot without clustering
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            sns.scatterplot(x='CP_1', y='CP_2', data=data_final_plot, ax=ax1)

            # Add annotations
            for i in range(len(data_final_plot)):
                ax1.text(
                    x=data_final_plot.loc[i, 'CP_1'], 
                    y=data_final_plot.loc[i, 'CP_2'] + 0.15, 
                    s=f"{data_final_plot.loc[i, 'Player']} ({data_final_plot.loc[i, 'Team']})", 
                    fontdict=dict(size=10)
                )

            ax1.set_title("PCA Analysis - Graphical representation of teams from the 5 analyzed leagues using 2 CPs")
            st.pyplot(fig1)

# KMEANS CLUSTERS
        with st.expander("K-Means Clustering"):
            nClusters = 4
            kmeans = KMeans(n_clusters=nClusters, random_state=SEED).fit(data_final)

            data_plot_kmeans = data_final.reset_index()
            data_plot_kmeans['Cluster'] = kmeans.labels_  # Añadir los labels de clusters a los datos

            # Create the plot
            fig2, ax2 = plt.subplots(figsize=(15, 8))
            sns.scatterplot(x='CP_1', y='CP_2', hue='Cluster', palette='viridis', data=data_plot_kmeans, ax=ax2)

            for i in range(len(data_plot_kmeans)):
                plt.text(
                    x=data_plot_kmeans.loc[i, 'CP_1'], 
                    y=data_plot_kmeans.loc[i, 'CP_2'] + 0.15, 
                    s=f"{data_plot_kmeans.loc[i, 'Player']} ({data_plot_kmeans.loc[i, 'Team']})",  # Usar 'Player' y 'Year'
                    fontdict=dict(size=10)
                )

            centroids = kmeans.cluster_centers_
            for i in range(len(centroids)):
                plt.plot(centroids[i][0], centroids[i][1], marker='+', color='firebrick', markersize=15)

            ax2.set_title("KMeans Clustering")
            st.pyplot(fig2)