import joblib
import pandas as pd
import numpy as np
import os

# --- PATH CONFIGURATION ---
# IMPORTANT: Adjust this absolute path to your project root if necessary.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'k-means_best_model.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_scaler.pkl')

# Re-define the feature extraction function (CRITICAL STEP)
def extract_burnout_features(df, time_window_points=8):
    """
    Extracts Slope, Peak Drop, and Initial Engagement features from new time-series data.
    """
    # CRITICAL: Ensure columns match the expected format from the original data
    # For YouTube Trending: Use views as engagement metric
    df = df.rename(columns={'trending_date': 'Observation_Date', 'video_id': 'Customer_ID', 'views': 'Engagement_Metric'})
    df['Observation_Date'] = pd.to_datetime(df['Observation_Date'], format='%y.%d.%m')
    
    # Sort and group
    df = df.sort_values(['Customer_ID', 'Observation_Date'])
    
    engagement_sequences = df.groupby('Customer_ID')['Engagement_Metric'].apply(
        lambda x: x.tail(time_window_points).values
    ).reset_index()
    
    feature_list = []
    
    for _, row in engagement_sequences.iterrows():
        customer_id = row['Customer_ID']
        sequence = row['Engagement_Metric']
        
        if len(sequence) < 2:
            continue

        time_points = np.arange(len(sequence))
        
        # 1. Gradual Decline Feature (Slope)
        slope = np.polyfit(time_points, sequence, 1)[0]
        
        # 2. Sudden Drop Feature (Peak Drop)
        peak_drop = np.max(sequence) - sequence[-1] 
        
        # 3. Initial Engagement 
        initial_engagement = np.mean(sequence[:min(3, len(sequence))])

        feature_list.append({
            'Customer_ID': customer_id,
            'Slope': slope,
            'Peak_Drop': peak_drop,
            'Initial_Engagement': initial_engagement
        })

    return pd.DataFrame(feature_list).set_index('Customer_ID')


def load_and_predict_new_data(new_raw_df):
    """Loads model/scaler, extracts features, scales, and predicts segments."""
    try:
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_model = joblib.load(MODEL_PATH)
    except FileNotFoundError as e:
        # NOTE: This error now indicates the file really doesn't exist, as paths are absolute/correct
        print(f"ERROR: Could not load model or scaler. Ensure you have run the notebook save cell (Cell 6). Details: {e}")
        return None

    # 1. Extract Features from raw data
    df_features = extract_burnout_features(new_raw_df, time_window_points=8)
    df_features = df_features.dropna()
    
    features = ['Slope', 'Peak_Drop', 'Initial_Engagement']
    X_new = df_features[features].copy()

    if X_new.empty:
        print("No complete sequences found for prediction.")
        return df_features
    
    # 2. Scale the features
    X_scaled = loaded_scaler.transform(X_new)

    # 3. Predict the cluster labels (Uses K-Means .predict() method)
    segments = loaded_model.predict(X_scaled)

    # 4. Attach results back to the original dataframe
    df_features['Segment'] = segments
    
    return df_features[['Segment', 'Slope', 'Peak_Drop', 'Initial_Engagement']]

# --- Example Usage ---
if __name__ == '__main__':
    print("Running inference script...")
    
    # Dummy data that models a "Slow Fade" segment (Segment 0)
    new_data_raw = pd.DataFrame({
        'trending_date': ['25.01.01', '25.01.02', '25.01.03', '25.01.04', 
                          '25.01.05', '25.01.06', '25.01.07', '25.01.08'],
        'video_id': ['A_SLOW_FADE', 'A_SLOW_FADE', 'A_SLOW_FADE', 'A_SLOW_FADE', 
                     'A_SLOW_FADE', 'A_SLOW_FADE', 'A_SLOW_FADE', 'A_SLOW_FADE'],
        'views': [10, 9, 8, 7, 6, 5, 4, 3] # Slow, steady decline
    })
    
    segmentation_results = load_and_predict_new_data(new_data_raw)
    
    if segmentation_results is not None:
        print("\n--- Segmentation Results for New Data ---")
        print(segmentation_results)