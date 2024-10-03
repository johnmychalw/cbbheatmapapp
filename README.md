**College Baseball Trackman Heatmaps Dashboard:**
This Streamlit app generates interactive heatmaps for analyzing Trackman Data. It allows users to visualize advanced metrics such as expected Batting Average (xBA), expected Slugging Percentage (xSLG), whiff rates, swing rates, contact rates, take rates, and pitch rates using trackman datasets.

<img width="500" alt="Screenshot 2024-10-03 at 9 31 49 AM" src="https://github.com/user-attachments/assets/9003821b-55ff-4178-bfb1-905913fdcc7d"> <img width="500" alt="Screenshot 2024-10-03 at 9 34 31 AM" src="https://github.com/user-attachments/assets/23f10254-83ba-4787-a02c-f5dc3e56f6bf">


**Features:**
Data Upload: Allows the user to upload their own Trackman CSV. 
Pitcher/Batter Scouting: This app allows the user to assess/scout both Pitchers and Hitters.
xBA Heatmap: Visualizes the expected batting average for a selected player based on pitch location.
xSLG Heatmap: Displays the expected slugging percentage for balls in play, showing how a player performs based on pitch location.
Whiff Rate Heatmap: Tracks swings and misses for a player in various pitch locations.
Swing Rate Heatmap: Shows how often a player swings at pitches in different zones.
Take Rate Heatmap: Visualizes the rate at which a player takes (does not swing at) pitches in the strike zone.
Contact Rate Heatmap: Visualizes where the player made any type of contact (including Foul Balls)
Pitch Rate Heatmap: Displays the distribution of pitch types thrown to a player.

Note: All Heatmaps are from the Pitcher's Perspective

Filter by Date: Select data from a specific date period for analysis.
Advanced Filters: Filter data by pitch types, pitcher hand (RHP or LHP), and various pitch counts (e.g., 0-0, 3-2).
KDE Heatmaps: The app uses kernel density estimation (KDE) to generate smooth and informative heatmaps.
Custom Models: xBA and xSLG visualizations using ML models from historical college trackman data.

**Models:**
xBA Model: Predicts the expected batting average based on the exit velocity and launch angle.
xSLG Model (knn_model): Predicts expected slugging percentage based on exit velocity, launch angle, and spray angle.

**Website:**
https://cbbheatmapapp.streamlit.app/ 

This app currently has a password, feel free to reach out to me at johnmychalwarren@outlook.com to request it.
