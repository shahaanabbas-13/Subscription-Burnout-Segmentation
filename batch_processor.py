import pandas as pd
import os
from datetime import datetime

import sys # <-- NEW IMPORT

# --- Add src/ to the system path for module finding ---
# This line tells Python to look in the subdirectory 'src'
sys.path.append('src') # <-- NEW LINE


from inference import load_and_predict_new_data # Import the core function

# --- FILE PATHS ---
# Define where the new data comes from and where the report goes
DATA_DIR = 'Data/archive(1)/' # <-- UPDATED: Use the new dataset in archive(1)
INPUT_FILE = DATA_DIR + 'USvideos.csv' # The new dataset file
REPORT_DIR = 'reports/' # <-- Optional: Change '../reports/' to 'reports/' for consistency

def run_daily_inference():
    """
    Simulates a scheduled job that loads the latest data, runs inference,
    and saves a high-risk churn report.
    """
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        
    print(f"--- Running Burnout Inference at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # 1. Load the latest raw data
    try:
        # NOTE: You would typically load ONLY the data collected since the last run
        df_raw_latest = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded {len(df_raw_latest)} rows for analysis.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE}. Exiting.")
        return

    # 2. Run the segmentation pipeline
    # This calls the logic from your inference.py file
    segmentation_results = load_and_predict_new_data(df_raw_latest)

    if segmentation_results is None or segmentation_results.empty:
        print("Inference completed, but no complete sequences were found for scoring.")
        return

    # 3. Create High-Risk Churn Report (Focus on Segment 0)
    # Assuming Segment 0 is "The Flash Burners" (High Risk)
    high_risk_customers = segmentation_results[segmentation_results['Segment'] == 0]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'high_risk_churn_list_{timestamp}.csv'
    report_path = os.path.join(REPORT_DIR, report_filename)
    
    high_risk_customers.to_csv(report_path, columns=['Segment', 'Slope', 'Peak_Drop'], index=True)
    
    print(f"\n✅ SUCCESS: Found {len(high_risk_customers)} high-risk customers.")
    print(f"Report saved to: {report_path}")
    
    # 4. (Business Action): Trigger notifications or marketing campaigns here
    # E.g., print(f"Triggering intervention for {high_risk_customers.index.tolist()}")


if __name__ == '__main__':
    run_daily_inference()