import pandas as pd
import os
from datetime import datetime

INPUT_FILE = "GEO.csv"

def clean_and_align_data():
    
    print("="*70)
    print("DATA CLEANING & ALIGNMENT")
    print("="*70)
    print()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "DATA")
    cleaning_dir = os.path.join(script_dir, "pr sample")
    
    master_file = os.path.join(cleaning_dir, "master_time_span.csv")
    input_file_path = os.path.join(data_dir, INPUT_FILE)
    output_file_path = os.path.join(cleaning_dir, INPUT_FILE)
    
    print(f"Input file: {INPUT_FILE}")
    print(f"Master skeleton: {master_file}")
    print(f"Output location: {output_file_path}")
    print()
    
    if not os.path.exists(master_file):
        print(f"✗ Error: Master time span file not found at {master_file}")
        print(f"  Please run generate_master_timespan.py first!")
        return
    
    if not os.path.exists(input_file_path):
        print(f"✗ Error: Input file not found at {input_file_path}")
        print(f"  Please ensure {INPUT_FILE} exists in the DATA directory")
        return
    
    print("✓ All required files found")
    print()
    
    print("Loading master time span skeleton...")
    master_df = pd.read_csv(master_file)
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    print(f"✓ Master skeleton loaded: {len(master_df)} dates")
    print(f"  Range: {master_df['Date'].min()} to {master_df['Date'].max()}")
    print()
    
    print(f"Loading input data from {INPUT_FILE}...")
    try:
        data_df = None
        file_ext = os.path.splitext(INPUT_FILE)[1].lower()
        
        if file_ext in ['.xls', '.xlsx']:
            print("  Detected Excel file format")
            data_df = pd.read_excel(input_file_path)
        elif INPUT_FILE.lower().endswith('.csv') or file_ext == '.csv':
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    print(f"  Trying encoding: {encoding}")
                    data_df = pd.read_csv(input_file_path, encoding=encoding)
                    print(f"  ✓ Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if encoding == encodings[0]:
                        try:
                            print(f"  CSV failed, trying as Excel file...")
                            data_df = pd.read_excel(input_file_path)
                            print(f"  ✓ Successfully loaded as Excel file")
                            break
                        except:
                            continue
            
            if data_df is None:
                print(f"✗ Error: Could not load file with any encoding")
                return
        else:
            print(f"✗ Error: Unsupported file format: {file_ext}")
            return
        
        if 'Price' in data_df.columns:
            if file_ext in ['.xls', '.xlsx']:
                data_df = pd.read_excel(input_file_path, skiprows=[1, 2])
            else:
                for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']:
                    try:
                        data_df = pd.read_csv(input_file_path, skiprows=[1, 2], encoding=encoding)
                        break
                    except:
                        continue
            data_df = data_df.rename(columns={'Price': 'Date'})
        
        if 'Date' not in data_df.columns:
            possible_date_cols = ['date', 'DATE', 'Time', 'time', 'Timestamp', 'observation_date', 'OBSERVATION_DATE', 'month', 'Month', 'MONTH']
            for col in possible_date_cols:
                if col in data_df.columns:
                    data_df = data_df.rename(columns={col: 'Date'})
                    break
        
        if 'Date' not in data_df.columns:
            print(f"✗ Error: No 'Date' column found in {INPUT_FILE}")
            print(f"  Available columns: {list(data_df.columns)}")
            return
        
        data_df = data_df[data_df['Date'].notna()]
        data_df = data_df[data_df['Date'] != '']
        
        data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
        data_df = data_df[data_df['Date'].notna()]
        
        data_cols = [col for col in data_df.columns if col != 'Date']
        data_df = data_df.dropna(how='all', subset=data_cols)
        
        print(f"✓ Input data loaded: {len(data_df)} rows")
        print(f"  Columns: {len(data_df.columns)} columns")
        print(f"  Date range: {data_df['Date'].min()} to {data_df['Date'].max()}")
        print()
        
    except Exception as e:
        print(f"✗ Error loading input file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Filtering dates...")
    master_start = master_df['Date'].min()
    master_end = master_df['Date'].max()
    
    original_count = len(data_df)
    data_df = data_df[(data_df['Date'] >= master_start) & (data_df['Date'] <= master_end)]
    dropped_count = original_count - len(data_df)
    
    print(f"✓ Filtered to master skeleton range")
    print(f"  Kept: {len(data_df)} rows")
    print(f"  Dropped: {dropped_count} rows (outside {master_start.date()} to {master_end.date()})")
    print()
    
    print("Merging with master skeleton...")
    merged_df = master_df.merge(data_df, on='Date', how='left')
    
    missing_count = merged_df.iloc[:, 1:].isna().any(axis=1).sum()
    print(f"✓ Merged successfully")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Rows with missing data: {missing_count}")
    print()
    
    print("Forward filling missing values...")
    data_columns = [col for col in merged_df.columns if col != 'Date']
    
    for col in data_columns:
        merged_df[col] = merged_df[col].ffill()
    
    remaining_na = merged_df[data_columns].isna().sum().sum()
    
    print(f"✓ Forward fill complete")
    if remaining_na > 0:
        print(f"  Warning: {remaining_na} values still missing (likely at start of dataset)")
        print(f"  These will remain as NaN")
    else:
        print(f"  All missing values filled successfully")
    print()
    
    print(f"Saving cleaned data to {output_file_path}...")
    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    merged_df.to_csv(output_file_path, index=False)
    
    print(f"✓ Saved successfully")
    print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input file: {INPUT_FILE}")
    print(f"Original data rows: {original_count}")
    print(f"After filtering: {len(data_df)}")
    print(f"Output rows (with skeleton): {len(merged_df)}")
    print(f"Missing dates added: {len(merged_df) - len(data_df)}")
    print(f"Output file: {output_file_path}")
    print()
    print("✓ DATA CLEANING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    clean_and_align_data()