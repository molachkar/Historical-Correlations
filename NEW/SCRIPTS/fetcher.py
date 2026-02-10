import yfinance as yf
import pandas as pd
from datetime import datetime
import os

print("="*70)
print("INDEPENDENT MARKET DATA EXTRACTION")
print("="*70)

instruments = {
    'XAUUSD': 'GC=F',
    'DXY': 'DX-Y.NYB',
    'NASDAQ': '^IXIC',
    'SP500': '^GSPC',
    'RUSSELL2000': '^RUT',
    'VIX': '^VIX',
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X'
}

start_date = '2010-01-01'
end_date = '2025-02-01'

output_dir = 'market_data_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\nCreated directory: {output_dir}\n")

successful = []
failed = []

for name, ticker in instruments.items():
    print(f"Processing {name} ({ticker})...")
    print("-" * 70)
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)
        
        if df.empty:
            print(f"  FAILED: No data returned for {name}")
            failed.append(name)
            print()
            continue
        
        actual_start = df.index[0].strftime('%Y-%m-%d')
        actual_end = df.index[-1].strftime('%Y-%m-%d')
        total_days = len(df)
        
        filename = f'{output_dir}/{name}.csv'
        df.to_csv(filename)
        
        print(f"  SUCCESS")
        print(f"     Data Range: {actual_start} to {actual_end}")
        print(f"     Total Days: {total_days}")
        print(f"     Columns: {list(df.columns)}")
        print(f"     File Saved: {filename}")
        print(f"     File Size: {os.path.getsize(filename) / 1024:.2f} KB")
        
        print(f"\n     First 3 rows:")
        print(df.head(3).to_string())
        
        successful.append({
            'name': name,
            'ticker': ticker,
            'start': actual_start,
            'end': actual_end,
            'days': total_days,
            'file': filename
        })
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        failed.append(name)
    
    print()

print("="*70)
print("EXTRACTION SUMMARY")
print("="*70)

if successful:
    print(f"\nSuccessfully extracted {len(successful)} instruments:\n")
    summary_df = pd.DataFrame(successful)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{output_dir}/SUMMARY.csv', index=False)
    print(f"\nSummary saved to: {output_dir}/SUMMARY.csv")

if failed:
    print(f"\nFailed to extract {len(failed)} instruments:")
    for instrument in failed:
        print(f"   - {instrument}")

print("\n" + "="*70)
print(f"ALL FILES SAVED IN: {output_dir}/")
print("="*70)

print("\nGenerated files:")
for file in os.listdir(output_dir):
    filepath = os.path.join(output_dir, file)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  - {file} ({size_kb:.2f} KB)")

print("\nEXTRACTION COMPLETE")