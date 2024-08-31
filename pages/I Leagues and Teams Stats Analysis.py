import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

file_path = '/Users/cristhianbeltran/Desktop/Sports_Data_Campus/Master/Proyecto_Final/Streamlit/data/league_stats_final.csv'

df = pd.read_csv(file_path)
df = df.drop(columns=['Unnamed: 0'])
df.reset_index(drop=True, inplace=True)

st.title("Leagues and Teams")
st.write('Squad Standard Stats FBref - Select Leagues and Seasons on the Side Bar')
st.markdown("__________")


with st.sidebar.expander("Select by League and Year"):
    comp = df['league'].unique().tolist()
    comp_selected = st.multiselect('League', options = comp, default=comp)

    years = df['year'].unique().tolist()
    years_selected = st.multiselect('Year', options=years, default=years)
    

df_filtered = df[df['league'].isin(comp_selected) & df['year'].isin(years_selected)]

st.dataframe(df_filtered)

#Navigation Menu
st.markdown("### **Select Analysis Category**")
category = st.selectbox("", ["Select a Category", "Goals and xG", "Shots", "Possession", "Clusters"])

# Display content based on category selection
if category == "Select a Category":
    st.warning("Please select a category from the dropdown above to view the analysis.")

# Goals and xG Analysis
elif category == "Goals and xG":
    st.subheader("Goals and xG Analysis")
# Average Goal Per Match
    with st.expander("View Average Goals per Match Over Time by League"):
    
        league_year = df_filtered.groupby(['league', 'year']).agg({
        'Gls': 'sum', 
        'MP': 'sum'
        }).reset_index()
    
        league_year['Gls_per_match'] = league_year['Gls'] / league_year['MP']
    
        plt.figure(figsize=(15, 8))
        sns.lineplot(x='year', y='Gls_per_match', hue='league', data=league_year, marker='o', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Average Goals per Match')
        plt.legend(title='League')
        st.pyplot(plt)
    
    
    # Average xG Per Match
    with st.expander("View xG per Match Over Time by League"):
    
        league_year = df_filtered.groupby(['league', 'year']).agg({
        'xG': 'sum', 
        'MP': 'sum'
        }).reset_index()
    
    
        league_year['xG_per_match'] = league_year['xG'] / league_year['MP']
    
        plt.figure(figsize=(15, 8))
    
    
        sns.lineplot(x='year', y='xG_per_match', hue='league', data=league_year, marker='o', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Average xG per Match')
        plt.legend(title='League')
        st.pyplot(plt)
    
    # Goals and xG per Match Across Leagues
    
    with st.expander("View Goals vs xG per Match by League"):
    
        league_year= df_filtered.groupby(['league', 'year']).agg({
            'Gls': 'sum', 
            'xG': 'sum', 
            'MP': 'sum'
        }).reset_index()
    
    
        league_year['Goals per match'] = league_year['Gls'] / league_year['MP']
        league_year['xG per match'] = league_year['xG'] / league_year['MP']
    
    
        plt.figure(figsize=(15, 8))
    
    
        sns.scatterplot(x='Goals per match', y='xG per match', hue='league', style='league', data=league_year, s=100)
    
    
        for i in range(league_year.shape[0]):
            plt.text(x=league_year['Goals per match'].iloc[i] + 0.01, 
                     y=league_year['xG per match'].iloc[i], 
                     s=f"{league_year['league'].iloc[i]} ({league_year['year'].iloc[i]})", 
                     fontsize=9)
    
    
        plt.xlabel('Goals per match')
        plt.ylabel('xG per match')
        plt.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
        plt.legend(title='League', loc='best')
        st.pyplot(plt)
    
    #Goals vs xG per Match by Team
    
    with st.expander("View Goals vs xG per Match by Team"):
    
        team_year= df_filtered.groupby(['league', 'year', 'Squad']).agg({
                'Gls': 'sum', 
                'xG': 'sum', 
                'MP': 'sum'
            }).reset_index()
    
        team_year['Goals per match'] = team_year['Gls'] / team_year['MP']
        team_year['xG per match'] = team_year['xG'] / team_year['MP']
        
        plt.figure(figsize=(15, 8))
        
        
        sns.scatterplot(x='Goals per match', y='xG per match', hue='league', style='league', data=team_year, s=100)
        
        
        for i in range(team_year.shape[0]):
            plt.text(x=team_year['Goals per match'].iloc[i] + 0.01, 
                     y=team_year['xG per match'].iloc[i], 
                     s=f"{team_year['Squad'].iloc[i]} ({team_year['year'].iloc[i]})", 
                     fontsize=9)
    
    
        plt.title('Goals vs xG per Match by Team')
        plt.xlabel('Goals per match')
        plt.ylabel('xG per match')
        plt.axline((0, 0), slope=1, linestyle='--', color='grey')  # line y=x for reference
        plt.legend(title='League', loc='best')
        st.pyplot(plt)
    
# Shots Analysis
elif category == "Shots":
    st.subheader("Shots Analysis")

# Shots per Match vs. Average Shot Quality (xG/Shot) by League and Year
    with st.expander("Shots per Match vs. Average Shot Quality (xG/Shot) by League and Year"):
        df_filtered['xG_per_shot'] = df_filtered['xG'] / df_filtered['Sh']  # Update this line to use df_filtered

        try:
            league_year_avg = df_filtered.groupby(['league', 'year']).agg({
                'Sh/90': 'mean',
                'xG_per_shot': 'mean',  # Now we can use the average xG per shot
                'country': 'first'  # Keep the country information
            }).reset_index()

        except KeyError as e:
            st.error(f"KeyError: {e}. Please check if the columns 'Sh/90', 'xG', or 'Sh' exist in your DataFrame.")
            st.stop()

        # Step 3: Plot the aggregated data
        plt.figure(figsize=(15, 8))
        
        sns.scatterplot(
            data=league_year_avg,
            x='Sh/90',  # Shots per match
            y='xG_per_shot',  # xG per shot
            hue='league',  # Use league for color coding
            style='league',  # Use league for different markers (optional)
            s=100,  # Size of the points
            palette={"liga_argentina": "blue", "brasileirao": "green", "eredivisie": "orange", "liga_mx": "red", "primeira_liga": "purple"}  
        )
        
        # Step 4: Add text annotations for each league-year combination
        for i in range(len(league_year_avg)):
            plt.text(
                league_year_avg['Sh/90'].iloc[i] + 0.1,  # Offset x position slightly to avoid overlap
                league_year_avg['xG_per_shot'].iloc[i],
                f"{league_year_avg['league'].iloc[i]} ({league_year_avg['year'].iloc[i]})",
                fontsize=9
            )
        
        # Add titles and labels
        plt.title('Shots per Match vs. Average Shot Quality (xG/Shot) by League and Year')
        plt.xlabel('Shots per Match')
        plt.ylabel('Average Shot Quality (xG/Shot)')
        
        # Customize legend
        plt.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)


# Shots per Match vs. Average Shot Quality (xG/Shot) by TEAM and Year
    with st.expander("Shots per Match vs. Average Shot Quality (xG/Shot) by League"):
        df_filtered['xG_per_shot'] = df_filtered['xG'] / df_filtered['Sh']  # Update this line to use df_filtered

        try:
            league_year_avg = df_filtered.groupby(['league', 'year']).agg({
                'Sh/90': 'mean',
                'xG_per_shot': 'mean',  # Now we can use the average xG per shot
                'Squad': 'first'  # Keep the country information
            }).reset_index()

        except KeyError as e:
            st.error(f"KeyError: {e}. Please check if the columns 'Sh/90', 'xG', or 'Sh' exist in your DataFrame.")
            st.stop()

        # Step 3: Plot the aggregated data
        plt.figure(figsize=(15, 8))
        
        sns.scatterplot(
            data=league_year_avg,
            x='Sh/90',  # Shots per match
            y='xG_per_shot',  # xG per shot
            hue='league',  # Use league for color coding
            style='league',  # Use league for different markers (optional)
            s=100,  # Size of the points
            palette={"liga_argentina": "blue", "brasileirao": "green", "eredivisie": "orange", "liga_mx": "red", "primeira_liga": "purple"}  
        )

        plt.xlim(10, None)
        
        # Step 4: Add text annotations for each league-year combination
        for i in range(len(league_year_avg)):
            plt.text(
                league_year_avg['Sh/90'].iloc[i] + 0.1,  # Offset x position slightly to avoid overlap
                league_year_avg['xG_per_shot'].iloc[i],
                f"{league_year_avg['Squad'].iloc[i]} ({league_year_avg['year'].iloc[i]})",
                fontsize=9
            )
        
        # Add titles and labels
        plt.title('Shots per Match vs. Average Shot Quality (xG/Shot) by Team')
        plt.xlabel('Shots per Match')
        plt.ylabel('Average Shot Quality (xG/Shot)')
        
        # Customize legend
        plt.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)

# Possesion Analysis
elif category == "Possession":
    st.subheader("Possession Analysis")

#Distribution of Ball Possession by League
    with st.expander("Distribution of Ball Possession by League"):
        league_year_avg= df_filtered.groupby(['league', 'year', 'Squad']).agg({
                'Poss': 'mean', 
            }).reset_index()

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='league', y='Poss', data=league_year_avg)
        plt.title('Distribution of Ball Possession by League')
        plt.ylabel('Ball Possession (%)')
        plt.xlabel('League')
        plt.xticks(rotation=45)
        st.pyplot(plt)

#Progressive Carries vs. Progressive Passes by League and Year
    with st.expander("Progressive Carries vs. Progressive Passes by League and Year"):
        league_year_avg= df_filtered.groupby(['league', 'year']).agg({
                'PrgC': 'mean',  # Average Progressive Carries
                'PrgP': 'mean',  # Average Progressive Passes
                'country': 'first'  # Keep the country information
        }).reset_index() 
        plt.figure(figsize=(14, 8))
        
        sns.scatterplot(
            data=league_year_avg,
            x='PrgC',  # Progressive Carries
            y='PrgP',  # Progressive Passes
            hue='league',  # Use league for color coding
            style='league',  # Use league for different markers (optional)
            s=100,  # Size of the points
            palette={"liga_argentina": "blue", "brasileirao": "green", "eredivisie": "orange", "liga_mx": "red", "primeira_liga": "purple"}  # Custom colors for each country
        )
        
        # Step 3: Add text annotations for each league-year combination
        for i in range(len(league_year_avg)):
            plt.text(
                league_year_avg['PrgC'].iloc[i] + 0.1,  # Offset x position slightly to avoid overlap
                league_year_avg['PrgP'].iloc[i],
                f"{league_year_avg['league'].iloc[i]} ({league_year_avg['year'].iloc[i]})",
                fontsize=9
            )
        
        # Add titles and labels
        plt.title('Progressive Carries vs. Progressive Passes by League and Year')
        plt.xlabel('Progressive Carries')
        plt.ylabel('Progressive Passes')
        
        # Customize legend
        plt.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)

#Progressive Carries vs. Progressive Passes by Team and Year
    with st.expander("Progressive Carries vs. Progressive Passes by Team and Year"):
        league_year_avg= df_filtered.groupby(['league', 'year', 'Squad']).agg({
                'PrgC': 'mean',  # Average Progressive Carries
                'PrgP': 'mean',  # Average Progressive Passes
                'country': 'first'  # Keep the country information
        }).reset_index() 
        plt.figure(figsize=(14, 8))
        
        sns.scatterplot(
            data=league_year_avg,
            x='PrgC',  # Progressive Carries
            y='PrgP',  # Progressive Passes
            hue='league',  # Use league for color coding
            style='league',  # Use league for different markers (optional)
            s=100,  # Size of the points
            palette={"liga_argentina": "blue", "brasileirao": "green", "eredivisie": "orange", "liga_mx": "red", "primeira_liga": "purple"}  # Custom colors for each country
        )
        
        # Step 3: Add text annotations for each league-year combination
        for i in range(len(league_year_avg)):
            plt.text(
                league_year_avg['PrgC'].iloc[i] + 0.1,  # Offset x position slightly to avoid overlap
                league_year_avg['PrgP'].iloc[i],
                f"{league_year_avg['Squad'].iloc[i]} ({league_year_avg['year'].iloc[i]})",
                fontsize=9
            )
        
        # Add titles and labels
        plt.title('Progressive Carries vs. Progressive Passes by League and Year')
        plt.xlabel('Progressive Carries')
        plt.ylabel('Progressive Passes')
        
        # Customize legend
        plt.legend(title='League', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)
        
#Correlation Between Progressive Carries, Passes, Shots, and Goals by League
    with st.expander("Correlation Between Progressive Carries, Passes, Shots, and Goals by League"):
        leagues = df_filtered['league'].unique()
        
        # Set up the figure grid
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))  # Adjust the grid size based on the number of leagues
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        
        # Iterate over leagues and create a heatmap for each one
        for i, league in enumerate(leagues):
            league_data = df_filtered[df_filtered['league'] == league]
            corr_matrix = league_data[['PrgC', 'PrgP', 'Poss', 'Gls']].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=axes[i])
            axes[i].set_title(f'{league} Correlation')
        
        # Adjust layout to make room for titles and labels
        plt.tight_layout()
        plt.suptitle('Correlation Between Progressive Carries, Passes, Possession, and Goals by League', y=1.02, fontsize=16)
        st.pyplot(plt)


# Clusters Analysis
elif category == "Clusters":
    st.subheader("Clusters Analysis - Teams")

    # Apply the filter using the sidebar selections
    df_filtered = df[df['league'].isin(comp_selected) & df['year'].isin(years_selected)]

    # Check if there's data to process after filtering
    if df_filtered.empty:
        st.warning("No data available for the selected leagues and years.")
    else:
        
# PCA
        with st.expander("PCA - Graphical representation of teams from the 5 analyzed leagues using 2 CPs"):
            off_metrics = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 
                           'PrgC', 'PrgP', 'Gls/90', 'Ast/90', 'G+A/90', 'xG/90', 'xAG/90', 
                           'xG+xAG', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT']
            data_selection = df_filtered[['Squad', 'league', 'year'] + off_metrics]
            data_preparation = data_selection.copy()
            data_preparation.set_index(['Squad', 'year'], inplace=True)
            data_preparation.drop(['league'], axis=1, inplace=True)

            # Standardize the data
            scaler = StandardScaler()
            data_preparation_scaled = scaler.fit_transform(data_preparation)

            # PCA process
            nOPT = 5
            pca = PCA(n_components=nOPT)
            new_scores = pca.fit_transform(data_preparation_scaled)
            data_final = pd.DataFrame(new_scores, index=data_preparation.index)  # Use the index from data_preparation
            data_final.columns = ["CP_" + str(i) for i in range(1, nOPT + 1)]

            # Prepare the final plot data
            data_final_plot = data_final.reset_index()  # Reset index to get 'Squad' and 'year' as columns

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
                    s=f"{data_final_plot.loc[i, 'Squad']} ({data_final_plot.loc[i, 'year']})", 
                    fontdict=dict(size=10)
                )

            ax1.set_title("PCA Analysis - Graphical representation of teams from the 5 analyzed leagues using 2 CPs")
            st.pyplot(fig1)

# KMEANS CLUSTERS
        with st.expander("K-Means Clustering"):
            # Perform KMeans clustering
            nClusters = 4
            kmeans = KMeans(n_clusters=nClusters, random_state=1234).fit(data_final)  # Ensure this line is executed before using `kmeans`

            data_plot_kmeans = data_final.reset_index()  # Use reset_index to get 'Squad' and 'year' columns
            data_plot_kmeans['Cluster'] = kmeans.labels_  # Add cluster labels from KMeans

            # Create the plot
            fig2, ax2 = plt.subplots(figsize=(15, 8))
            sns.scatterplot(x='CP_1', y='CP_2', hue='Cluster', palette='viridis', data=data_plot_kmeans, ax=ax2)

            # Add annotations for each point, including Squad and Year
            for i in range(len(data_plot_kmeans)):
                ax2.text(
                    x=data_plot_kmeans.loc[i, 'CP_1'], 
                    y=data_plot_kmeans.loc[i, 'CP_2'] + 0.15, 
                    s=f"{data_plot_kmeans.loc[i, 'Squad']} ({data_plot_kmeans.loc[i, 'year']})", 
                    fontdict=dict(size=10)
                )

            # Plot the centroids of the clusters
            centroids = kmeans.cluster_centers_
            for i in range(len(centroids)):
                ax2.plot(centroids[i][0], centroids[i][1], marker='+', color='red', markersize=10)

            # Add title to the plot
            ax2.set_title("KMeans Clustering")
            st.pyplot(fig2)
