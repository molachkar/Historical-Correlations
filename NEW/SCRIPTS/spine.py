import pandas as pd
import os
from datetime import datetime

def generate_master_timespan():
    """
    Generate master time span CSV file
    Daily dates from 2011-01-01 to 2025-01-01
    Output: CLEANING/master_time_span.csv
    """
    
    print("="*70)
    print("MASTER TIME SPAN GENERATOR")
    print("="*70)
    print()
    
    # Set paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "CLEANING")
    
    # Create CLEANING directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Generate date range
    start_date = '2011-01-01'
    end_date = '2025-01-01'
    
    print(f"✓ Generating daily dates from {start_date} to {end_date}")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range.strftime('%Y-%m-%d')
    })
    
    print(f"✓ Generated {len(df)} dates")
    print()
    
    # Save to CSV
    output_file = os.path.join(output_dir, "master_time_span.csv")
    df.to_csv(output_file, index=False)
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total dates: {len(df)}")
    print(f"Start date: {df['Date'].iloc[0]}")
    print(f"End date: {df['Date'].iloc[-1]}")
    print(f"Output file: {output_file}")
    print()
    print("✓ MASTER TIME SPAN CREATED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    generate_master_timespan()