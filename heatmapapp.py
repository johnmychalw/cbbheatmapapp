
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime

##### MAINNNNNNNN

# Load the saved models
xba_model = joblib.load('xbamodel.pkl')
knn_model = joblib.load('xSLG.pkl')  # xSLG model

# Streamlit App
st.title("HeatMaps Dashboard")

# Add a selection box for scouting type
scouting_type = st.selectbox("Select Scouting Type", ["Batter Scouting", "Pitching Scouting"])

# Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the CSV data
    new_data = pd.read_csv(uploaded_file)

    # Create a copy of the data to use for whiffs, swing, take, and pitch calculations (pitch-by-pitch data)
    pitch_by_pitch_data = new_data.copy()

    # Try to convert the 'Date' column to datetime format, allowing for mixed formats
    new_data['Date'] = pd.to_datetime(new_data['Date'], errors='coerce', infer_datetime_format=True)
    pitch_by_pitch_data['Date'] = pd.to_datetime(pitch_by_pitch_data['Date'], errors='coerce', infer_datetime_format=True)

    # Remove duplicates
    new_data = new_data.drop_duplicates()
    pitch_by_pitch_data = pitch_by_pitch_data.drop_duplicates()

    # Remove duplicates, assuming 'Date' and 'PitchNo' uniquely identify entries
    new_data = new_data.drop_duplicates(subset=['Date', 'PitchNo'])
    pitch_by_pitch_data = pitch_by_pitch_data.drop_duplicates(subset=['Date', 'PitchNo'])

    # Adjust based on scouting type
    if scouting_type == "Batter Scouting":
        team_column = "BatterTeam"
        handedness_column = "PitcherThrows"
    else:  # Pitching Scouting
        team_column = "PitcherTeam"
        handedness_column = "BatterSide"

    # Remove rows with missing values in key columns for the xBA and xSLG calculation
    new_data.dropna(subset=['Angle', 'ExitSpeed', 'Direction'], inplace=True)

    # Create a copy of the data to be used for the heatmaps
    player_data = new_data.copy()


    ### DATE RANGE SELECTION ###
    # Default date range to the minimum and maximum dates in the dataset
    min_date = player_data['Date'].min()
    max_date = player_data['Date'].max()

    # Create two columns for side-by-side selection
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input('Select Start Date', value=min_date, min_value=min_date, max_value=max_date)

    with col2:
        end_date = st.date_input('Select End Date', value=max_date, min_value=min_date, max_value=max_date)

    # Filter both dataframes by the selected date range
    player_data = player_data[(player_data['Date'] >= pd.to_datetime(start_date)) & (player_data['Date'] <= pd.to_datetime(end_date))]
    pitch_by_pitch_data = pitch_by_pitch_data[(pitch_by_pitch_data['Date'] >= pd.to_datetime(start_date)) & (pitch_by_pitch_data['Date'] <= pd.to_datetime(end_date))]

    ### PLAYER SELECTION ###
    player_column = 'Batter' if scouting_type == "Batter Scouting" else 'Pitcher'
    player = st.selectbox(f'Select a {player_column}:', player_data[player_column].unique())

    # Filter both dataframes for the selected player
    player_data = player_data[player_data[player_column] == player].copy()
    pitch_by_pitch_data = pitch_by_pitch_data[pitch_by_pitch_data[player_column] == player].copy()

    # Sort the data by date (chronologically by at-bat)
    player_data = player_data.sort_values('Date')
    pitch_by_pitch_data = pitch_by_pitch_data.sort_values('Date')

    # Reset index to treat each at-bat sequentially
    player_data.reset_index(drop=True, inplace=True)
    pitch_by_pitch_data.reset_index(drop=True, inplace=True)

    ### HEATMAP SELECTION ###
    heatmap_options = ['xBA', 'xSLG', 'Contact', 'Whiff', 'Swing', 'Take', 'Pitch']
    selected_heatmap = st.selectbox('Select HeatMap', heatmap_options)


    ### xBA Prediction (For xBA HeatMap) ###
    if selected_heatmap == 'xBA':
        # Set predicted_probability to 0 by default (covers strikeouts and initializes non-strikeouts)
        player_data['predicted_probability'] = 0.0

        # Filter for non-strikeout data and calculate probabilities
        non_strikeouts = player_data[player_data['KorBB'] != 'Strikeout'].copy()
        if not non_strikeouts.empty:
            probabilities = xba_model.predict_proba(non_strikeouts[['Angle', 'ExitSpeed']])[:, 1]
            non_strikeouts['predicted_probability'] = probabilities
            # Combine non-strikeout predicted probabilities back to the main dataframe
            player_data.update(non_strikeouts[['predicted_probability']])

    ### xSLG Prediction (For xSLG HeatMap) ###
    elif selected_heatmap == 'xSLG':
        # Set predicted_total_bases to 0 by default (covers strikeouts and initializes non-strikeouts)
        player_data['predicted_total_bases'] = 0

        # Filter for non-strikeout data and calculate predicted total bases
        non_strikeouts = player_data[player_data['KorBB'] != 'Strikeout'].copy()
        if not non_strikeouts.empty:
            # Get the class predictions (not probabilities)
            hit_predictions = knn_model.predict(non_strikeouts[['Angle', 'ExitSpeed', 'Direction']])
            # Calculate the predicted total bases (0 for out, 1 for single, 2 for double, etc.)
            non_strikeouts['predicted_total_bases'] = hit_predictions
            # Combine non-strikeout predicted total bases back to the main dataframe
            player_data.update(non_strikeouts[['predicted_total_bases']])

    ### WHIFFS CALCULATION (For Whiffs HeatMap) ###
    elif selected_heatmap == 'Whiff':
        # Add a 'whiff' column where 1 = whiff (PitchCall = StrikeSwinging) and 0 otherwise
        pitch_by_pitch_data['whiff'] = pitch_by_pitch_data.apply(lambda row: 1 if row['PitchCall'] == 'StrikeSwinging' else 0, axis=1)
    
    ### SWING% CALCULATION (For Swing HeatMap) ###
    elif selected_heatmap == 'Swing':
        # Add a 'swing' column where 1 = swing (various PitchCalls) and 0 otherwise
        swing_conditions = ['StrikeSwinging', 'InPlay', 'FoulBall', 'FoulBallFieldable', 'FoulBallNotFieldable']
        pitch_by_pitch_data['swing'] = pitch_by_pitch_data.apply(lambda row: 1 if row['PitchCall'] in swing_conditions else 0, axis=1)

    ### TAKE% CALCULATION (For Take HeatMap) ###
    elif selected_heatmap == 'Take':
        # Add a 'take' column where 1 = take (PitchCall = 'BallCalled' or 'StrikeCalled') and 0 otherwise
        take_conditions = ['BallCalled', 'StrikeCalled']
        pitch_by_pitch_data['take'] = pitch_by_pitch_data.apply(lambda row: 1 if row['PitchCall'] in take_conditions else 0, axis=1)

    ### PITCH% CALCULATION (For Pitch HeatMap) ###
    elif selected_heatmap == 'Pitch':
        # Add a 'pitch' column where every row is 1 (since all rows are valid for pitch percentage)
        pitch_by_pitch_data['pitch'] = 1

    elif selected_heatmap == 'Contact':
    # Add a 'contact' column where 1 = contact (PitchCall is 'InPlay', 'FoulBall', 'FoulBallFieldable', 'FoulBallNotFieldable') and 0 otherwise
        contact_conditions = ['InPlay', 'FoulBall', 'FoulBallFieldable', 'FoulBallNotFieldable']
        pitch_by_pitch_data['contact'] = pitch_by_pitch_data.apply(lambda row: 1 if row['PitchCall'] in contact_conditions else 0, axis=1)


    # Create three columns for side-by-side selection of filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Handedness filter
        handedness_options = [' ', 'vs Right', 'vs Left']
        selected_handedness = st.selectbox('Select Opponent Handedness', handedness_options)
    
    with col2:
        # Pitch filter
        pitch_filter_options = [' ', 'Fastballs', 'Breaking Balls', 'Change-Ups']
        selected_filter = st.selectbox('Select Pitch', pitch_filter_options)
    
    with col3:
        # Count filter
        count_options = [' ', '0-0', '1-0', '2-0', '3-0', '0-1', '0-2', '1-1', '2-1', '3-1', '1-2', '2-2', '3-2']
        selected_count = st.selectbox('Select Count', count_options)
    
    # Apply filters
    if selected_heatmap in ['xBA', 'xSLG']:
        filtered_data = player_data
    else:
        filtered_data = pitch_by_pitch_data
    
    # Apply handedness filter
    if selected_handedness != ' ':
        filtered_data = filtered_data[filtered_data[handedness_column] == selected_handedness.split()[-1]]
    
    # Apply pitch filter
    if selected_filter != ' ':
        if selected_filter == 'Fastballs':
            filtered_data = filtered_data[filtered_data['TaggedPitchType'].isin(['Fastball', 'Cutter', 'Sinker'])]
        elif selected_filter == 'Breaking Balls':
            filtered_data = filtered_data[filtered_data['TaggedPitchType'].isin(['Slider', 'Curveball'])]
        elif selected_filter == 'Change-Ups':
            filtered_data = filtered_data[filtered_data['TaggedPitchType'].isin(['ChangeUp', 'Changeup', 'Splitter'])]
    
    # Apply count filter
    if selected_count != ' ':
        balls, strikes = map(int, selected_count.split('-'))
        filtered_data = filtered_data[(filtered_data['Balls'] == balls) & (filtered_data['Strikes'] == strikes)]
    
    # Use filtered_data for heatmap generation
    in_play_data = filtered_data
    

    def plot_heatmap_for_batter(data, batter_name, value_column):
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Create a KDE plot using PlateLocSide and PlateLocHeight, weighted by the selected column
        sns.kdeplot(
            x=data['PlateLocSide'], 
            y=data['PlateLocHeight'], 
            fill=True, 
            cmap="coolwarm", 
            weights=data[value_column], 
            levels=100, 
            thresh=0,
            ax=ax
        )
        
        # Add a white rectangle to represent the strike zone
        strike_zone = patches.Rectangle((-0.85, 1.5), 1.7, 2.0, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(strike_zone)
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([1.0, 4.0])
        st.pyplot(fig)  # Use st.pyplot instead of plt.show()
    

    
    if not in_play_data.empty:
        handedness_label = " " if selected_handedness == " " else f"{selected_handedness}"
        
        if selected_heatmap == 'xBA':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>xBA Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'predicted_probability')
        elif selected_heatmap == 'Whiff':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>Whiff Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'whiff')
        elif selected_heatmap == 'Swing':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>Swing Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'swing')
        elif selected_heatmap == 'Take':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>Take Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'take')
        elif selected_heatmap == 'Pitch':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>Pitch Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'pitch')
        elif selected_heatmap == 'xSLG':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>xSLG Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'predicted_total_bases')
        # New case for 'Contact' HeatMap
        elif selected_heatmap == 'Contact':
            st.markdown(f"<h3 style='text-align: center; font-size:23px;'>Contact Heatmap for {player} [{handedness_label} | {selected_filter} | {selected_count}]</h3>", unsafe_allow_html=True)
            plot_heatmap_for_batter(in_play_data, player, 'contact')
        
    # Display pitch count at the bottom in small text
    pitch_count = len(in_play_data)
    st.markdown(f"<p style='font-size: 10px;'>PitchCount: {pitch_count}</p>", unsafe_allow_html=True)
