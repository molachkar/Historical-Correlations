import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    Comprehensive Technical Indicators Calculator
    Calculates all indicators that can be computed from OHLCV data only
    """
    
    def __init__(self, df):
        """
        Initialize with OHLCV dataframe
        df must have columns: Date, Open, High, Low, Close, Volume
        """
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
    # ==================== EMA CALCULATIONS ====================
    
    def calculate_ema(self, period, column='Close'):
        """Calculate Exponential Moving Average"""
        return self.df[column].ewm(span=period, adjust=False).mean()
    
    def calculate_all_emas(self):
        """Calculate EMA 9, 21, 50, 200"""
        self.df['EMA_9'] = self.calculate_ema(9)
        self.df['EMA_21'] = self.calculate_ema(21)
        self.df['EMA_50'] = self.calculate_ema(50)
        self.df['EMA_200'] = self.calculate_ema(200)
        
    # ==================== VOLATILITY INDICATORS ====================
    
    def calculate_parkinson_volatility(self, window=20):
        """
        Parkinson Volatility - Uses High/Low
        More efficient than close-to-close volatility
        """
        hl_ratio = np.log(self.df['High'] / self.df['Low'])
        parkinson = np.sqrt((1 / (4 * window * np.log(2))) * 
                           (hl_ratio ** 2).rolling(window=window).sum())
        return parkinson
    
    def calculate_yang_zhang_volatility(self, window=20):
        """
        Yang-Zhang Volatility - Uses OHLC
        Combines overnight and intraday volatility
        """
        # Overnight volatility
        ho = np.log(self.df['Open'] / self.df['Close'].shift(1))
        
        # Open-to-close volatility
        co = np.log(self.df['Close'] / self.df['Open'])
        
        # Rogers-Satchell volatility
        hl = np.log(self.df['High'] / self.df['Close'])
        lc = np.log(self.df['Low'] / self.df['Close'])
        oh = np.log(self.df['Open'] / self.df['High'])
        ol = np.log(self.df['Open'] / self.df['Low'])
        
        rs = (hl * (hl - co) + lc * (lc - co)).rolling(window=window).mean()
        
        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        overnight_var = ho.rolling(window=window).var()
        openclose_var = co.rolling(window=window).var()
        
        yang_zhang = np.sqrt(overnight_var + k * openclose_var + (1 - k) * rs)
        
        # Annualize (252 trading days)
        return yang_zhang * np.sqrt(252)
    
    def calculate_volatility_percentile(self, volatility, window=252):
        """Calculate percentile rank of current volatility"""
        def percentile_rank(series):
            return pd.Series(
                [np.sum(series.iloc[:i+1] < series.iloc[i]) / (i + 1) * 100 
                 if i > 0 else 50 
                 for i in range(len(series))],
                index=series.index
            )
        return percentile_rank(volatility)
    
    def classify_volatility_regime(self, percentile):
        """Classify volatility regime based on percentile"""
        return percentile.apply(lambda x: 
            'high' if x >= 66.67 else 
            'medium' if x >= 33.33 else 
            'low'
        )
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    # ==================== TREND INDICATORS ====================
    
    def calculate_adx(self, period=14):
        """Calculate ADX, +DI, -DI"""
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=self.df.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=self.df.index).rolling(window=period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm / atr)
        minus_di = 100 * (minus_dm / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    # ==================== BOLLINGER BANDS ====================
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = ((upper - lower) / middle) * 100
        return upper, middle, lower, width
    
    # ==================== ATR ====================
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    # ==================== STOCHASTIC ====================
    
    def calculate_stochastic(self, k_period=14, d_period=3):
        """Calculate Stochastic %K and %D"""
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()
        
        k = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    # ==================== ICHIMOKU CLOUD ====================
    
    def calculate_ichimoku(self, tenkan=9, kijun=26, senkou_b=52, displacement=26):
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line)
        tenkan_high = self.df['High'].rolling(window=tenkan).max()
        tenkan_low = self.df['Low'].rolling(window=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = self.df['High'].rolling(window=kijun).max()
        kijun_low = self.df['Low'].rolling(window=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = self.df['High'].rolling(window=senkou_b).max()
        senkou_b_low = self.df['Low'].rolling(window=senkou_b).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        return tenkan_sen, kijun_sen, senkou_a, senkou_b
    
    # ==================== ADVANCED INDICATORS ====================
    
    def calculate_vwap(self):
        """Calculate Volume Weighted Average Price"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        return (typical_price * self.df['Volume']).cumsum() / self.df['Volume'].cumsum()
    
    def calculate_parabolic_sar(self, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        
        psar = np.zeros(len(close))
        trend = np.ones(len(close))
        ep = np.zeros(len(close))
        af = np.zeros(len(close))
        
        # Initialize
        psar[0] = close[0]
        trend[0] = 1
        ep[0] = high[0]
        af[0] = af_start
        
        for i in range(1, len(close)):
            if trend[i-1] == 1:  # Uptrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                if low[i] < psar[i]:  # Trend reversal
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
                
                if high[i] > psar[i]:  # Trend reversal
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(psar, index=self.df.index)
    
    def calculate_awesome_oscillator(self, short=5, long=34):
        """Calculate Awesome Oscillator"""
        median = (self.df['High'] + self.df['Low']) / 2
        short_ma = median.rolling(window=short).mean()
        long_ma = median.rolling(window=long).mean()
        return short_ma - long_ma
    
    def calculate_williams_r(self, period=14):
        """Calculate Williams %R"""
        high_max = self.df['High'].rolling(window=period).max()
        low_min = self.df['Low'].rolling(window=period).min()
        return -100 * ((high_max - self.df['Close']) / (high_max - low_min))
    
    def calculate_cci(self, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mad)
    
    def calculate_mfi(self, period=14):
        """Calculate Money Flow Index"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def calculate_roc(self, period=12):
        """Calculate Rate of Change"""
        return ((self.df['Close'] - self.df['Close'].shift(period)) / 
                self.df['Close'].shift(period)) * 100
    

    
    def calculate_efficiency_ratio(self, period=10):
        """Calculate Efficiency Ratio (Kaufman)"""
        change = abs(self.df['Close'] - self.df['Close'].shift(period))
        volatility = abs(self.df['Close'].diff()).rolling(window=period).sum()
        return change / volatility
    
    # ==================== MASTER CALCULATION ====================
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        print("Calculating EMAs...")
        self.calculate_all_emas()
        
        print("Calculating Volatility Indicators...")
        self.df['Parkinson_Vol'] = self.calculate_parkinson_volatility()
        self.df['Yang_Zhang_Vol'] = self.calculate_yang_zhang_volatility()
        self.df['Vol_Percentile'] = self.calculate_volatility_percentile(self.df['Yang_Zhang_Vol'])
        self.df['Vol_Regime'] = self.classify_volatility_regime(self.df['Vol_Percentile'])
        
        print("Calculating Momentum Indicators...")
        self.df['RSI'] = self.calculate_rsi()
        macd, signal, hist = self.calculate_macd()
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = signal
        self.df['MACD_Hist'] = hist
        
        print("Calculating Trend Indicators...")
        adx, plus_di, minus_di = self.calculate_adx()
        self.df['ADX'] = adx
        self.df['Plus_DI'] = plus_di
        self.df['Minus_DI'] = minus_di
        
        print("Calculating Bollinger Bands...")
        bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bollinger_bands()
        self.df['BB_Upper'] = bb_upper
        self.df['BB_Middle'] = bb_middle
        self.df['BB_Lower'] = bb_lower
        self.df['BB_Width'] = bb_width
        
        print("Calculating ATR...")
        self.df['ATR'] = self.calculate_atr()
        
        print("Calculating Stochastic...")
        stoch_k, stoch_d = self.calculate_stochastic()
        self.df['Stoch_K'] = stoch_k
        self.df['Stoch_D'] = stoch_d
        
        print("Calculating Ichimoku Cloud...")
        tk, kj, sa, sb = self.calculate_ichimoku()
        self.df['Ichimoku_TK'] = tk
        self.df['Ichimoku_KJ'] = kj
        self.df['Ichimoku_SA'] = sa
        self.df['Ichimoku_SB'] = sb
        
        print("Calculating Advanced Indicators...")
        self.df['VWAP'] = self.calculate_vwap()
        self.df['PSAR'] = self.calculate_parabolic_sar()
        self.df['AO'] = self.calculate_awesome_oscillator()
        self.df['Williams_R'] = self.calculate_williams_r()
        self.df['CCI'] = self.calculate_cci()
        self.df['MFI'] = self.calculate_mfi()
        self.df['ROC'] = self.calculate_roc()
        
        
        print("Calculating Efficiency Ratio...")
        self.df['Efficiency_Ratio'] = self.calculate_efficiency_ratio()
        
        print("✓ All indicators calculated!")
        return self.df
    
    # ==================== OUTPUT FORMATTING ====================
    
    def format_to_deepin_structure(self, start_date='2011-01-01'):
        """
        Format data to match deepin_daily.json structure
        Only includes rows from start_date onwards to ensure valid indicators
        """
        # Filter to start_date onwards
        filtered_df = self.df[self.df['Date'] >= start_date].copy()
        
        print(f"Filtering data from {start_date} onwards...")
        print(f"  Rows before filtering: {len(self.df)}")
        print(f"  Rows after filtering: {len(filtered_df)}")
        
        results = []
        for idx, row in filtered_df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            
            results.append({
                "date": date_str,
                "indicators": {
                    "volatility": {
                        "parkinson": round(row['Parkinson_Vol'], 4) if pd.notna(row['Parkinson_Vol']) else None,
                        "yang_zhang": round(row['Yang_Zhang_Vol'], 4) if pd.notna(row['Yang_Zhang_Vol']) else None,
                        "percentile": round(row['Vol_Percentile'], 2) if pd.notna(row['Vol_Percentile']) else None,
                        "regime": row['Vol_Regime'] if pd.notna(row['Vol_Regime']) else None
                    },
                    "ema": {
                        "ema_9": round(row['EMA_9'], 2) if pd.notna(row['EMA_9']) else None,
                        "ema_21": round(row['EMA_21'], 2) if pd.notna(row['EMA_21']) else None,
                        "ema_50": round(row['EMA_50'], 2) if pd.notna(row['EMA_50']) else None,
                        "ema_200": round(row['EMA_200'], 2) if pd.notna(row['EMA_200']) else None
                    },
                    "momentum": {
                        "rsi": round(row['RSI'], 2) if pd.notna(row['RSI']) else None,
                        "macd": round(row['MACD'], 2) if pd.notna(row['MACD']) else None,
                        "sig": round(row['MACD_Signal'], 2) if pd.notna(row['MACD_Signal']) else None,
                        "hist": round(row['MACD_Hist'], 2) if pd.notna(row['MACD_Hist']) else None
                    },
                    "trend": {
                        "adx": round(row['ADX'], 2) if pd.notna(row['ADX']) else None,
                        "pos": round(row['Plus_DI'], 2) if pd.notna(row['Plus_DI']) else None,
                        "neg": round(row['Minus_DI'], 2) if pd.notna(row['Minus_DI']) else None
                    },
                    "bb": {
                        "upper": round(row['BB_Upper'], 2) if pd.notna(row['BB_Upper']) else None,
                        "mid": round(row['BB_Middle'], 2) if pd.notna(row['BB_Middle']) else None,
                        "lower": round(row['BB_Lower'], 2) if pd.notna(row['BB_Lower']) else None,
                        "width": round(row['BB_Width'], 2) if pd.notna(row['BB_Width']) else None
                    },
                    "vol": {
                        "atr": round(row['ATR'], 2) if pd.notna(row['ATR']) else None
                    },
                    "stoch": {
                        "k": round(row['Stoch_K'], 2) if pd.notna(row['Stoch_K']) else None,
                        "d": round(row['Stoch_D'], 2) if pd.notna(row['Stoch_D']) else None
                    },
                    "ichimoku": {
                        "tk": round(row['Ichimoku_TK'], 2) if pd.notna(row['Ichimoku_TK']) else None,
                        "kj": round(row['Ichimoku_KJ'], 2) if pd.notna(row['Ichimoku_KJ']) else None,
                        "sa": round(row['Ichimoku_SA'], 2) if pd.notna(row['Ichimoku_SA']) else None,
                        "sb": round(row['Ichimoku_SB'], 2) if pd.notna(row['Ichimoku_SB']) else None
                    },
                    "adv": {
                        "vwap": round(row['VWAP'], 2) if pd.notna(row['VWAP']) else None,
                        "psar": round(row['PSAR'], 2) if pd.notna(row['PSAR']) else None,
                        "ao": round(row['AO'], 2) if pd.notna(row['AO']) else None,
                        "willr": round(row['Williams_R'], 2) if pd.notna(row['Williams_R']) else None,
                        "cci": round(row['CCI'], 2) if pd.notna(row['CCI']) else None,
                        "mfi": round(row['MFI'], 2) if pd.notna(row['MFI']) else None,
                        "roc": round(row['ROC'], 2) if pd.notna(row['ROC']) else None
                    },
                    "microstructure": {
                        "spread_estimate": None,  # Not calculable from OHLCV
                        "efficiency_ratio": round(row['Efficiency_Ratio'], 4) if pd.notna(row['Efficiency_Ratio']) else None
                    }
                }
            })
        
        return results


def main():
    """Main execution function"""
    
    print("="*70)
    print("TECHNICAL INDICATORS CALCULATOR - OPTIMIZED")
    print("No row skipping - filters by date instead")
    print("="*70)
    print()
    
    # Set paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "DATA", "XAUUSD.csv")
    output_dir = os.path.join(script_dir, "DATA")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Script directory: {script_dir}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    if not os.path.exists(input_file):
        print(f"✗ Error: Input file not found at {input_file}")
        print(f"  Please ensure XAUUSD.csv is in the DATA subdirectory")
        return
    
    print(f"✓ Found input file")
    print(f"Loading data from {input_file}...")
    
    try:
        # Read the CSV - first row is headers (Price,Close,High,Low,Open,Volume)
        # Second row is ticker symbols, third row is empty 'Date' row
        df = pd.read_csv(input_file, skiprows=[1, 2])  # Skip rows with ticker and empty date
        
        # Rename 'Price' column to 'Date' since that's what it contains
        df = df.rename(columns={'Price': 'Date'})
        
        # Remove any rows with missing dates
        df = df[df['Date'].notna()]
        df = df[df['Date'] != '']
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Loaded {len(df)} rows of data")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print()
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("  Trying alternative loading method...")
        try:
            df = pd.read_csv(input_file)
            print(f"  Available columns: {list(df.columns)}")
            print(f"  First few rows:")
            print(df.head(10))
        except:
            pass
        return
    
    # Verify columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    print("✓ All required columns present")
    print()
    
    # Calculate indicators on ALL data (including 2010)
    print("="*70)
    print("CALCULATING INDICATORS ON FULL DATASET")
    print("="*70)
    print()
    
    calc = TechnicalIndicators(df)
    result_df = calc.calculate_all_indicators()
    
    print()
    print("="*70)
    print("FORMATTING OUTPUT")
    print("="*70)
    print()
    
    # Format to JSON structure starting from 2011-01-01
    # This ensures all indicators have proper lookback period
    print("Formatting to deepin_daily.json structure...")
    print("  (Filtering to 2011-01-01 onwards for clean data)")
    
    json_output = calc.format_to_deepin_structure(start_date='2011-01-01')
    
    # Create output structure
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "daily_data": json_output
    }
    
    # Save JSON
    output_json = os.path.join(output_dir, "xauusd_technical_indicators.json")
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ JSON saved to: {output_json}")
    print(f"  Total days: {len(json_output)}")
    
    # Save full CSV with all indicators (including 2010 data)
    output_csv = os.path.join(output_dir, "xauusd_all_indicators.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"✓ Full CSV saved to: {output_csv}")
    print(f"  (Includes all data from 2010 onwards)")
    
    # Save filtered CSV (2011 onwards only)
    output_csv_filtered = os.path.join(output_dir, "INDICATORS.csv")
    filtered_df = result_df[result_df['Date'] >= '2011-01-01'].copy()
    filtered_df.to_csv(output_csv_filtered, index=False)
    print(f"✓ Filtered CSV saved to: {output_csv_filtered}")
    print(f"  (2011-01-01 onwards only - for merging)")
    
    # Summary statistics
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total rows processed: {len(result_df)}")
    print(f"Rows in JSON output: {len(json_output)} (from 2011-01-01)")
    print(f"Date range in full data: {result_df['Date'].min()} to {result_df['Date'].max()}")
    print(f"Date range in filtered output: 2011-01-01 to {result_df['Date'].max()}")
    print()
    
    # Check for remaining NaN values in filtered dataset
    nan_counts = filtered_df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        print("⚠ Warning: NaN values in filtered dataset (2011+):")
        for col, count in nan_cols.items():
            print(f"  {col}: {count} NaN values")
        total_nans = nan_counts.sum()
        print(f"  Total NaN values: {total_nans}")
    else:
        print("✓ No NaN values in filtered dataset!")
    
    print()
    print("Calculated indicators:")
    print("  ✓ EMAs (9, 21, 50, 200)")
    print("  ✓ Volatility (Parkinson, Yang-Zhang, Regime)")
    print("  ✓ Momentum (RSI, MACD)")
    print("  ✓ Trend (ADX, +DI, -DI)")
    print("  ✓ Bollinger Bands")
    print("  ✓ ATR")
    print("  ✓ Stochastic (%K, %D)")
    print("  ✓ Ichimoku Cloud (TK, KJ, SA, SB)")
    print("  ✓ Advanced (PSAR, AO, Williams %R, CCI, MFI, ROC)")
    print("  ✓ VWAP, Efficiency Ratio")
    print()
    print("Skipped (not calculable from OHLCV):")
    print("  ✗ Volume Profile (POC, VAH, VAL)")
    print("  ✗ Accumulation Score")
    print("  ✗ Spread Estimate")
    print("  ✗ Probability Metrics")
    print()
    print("="*70)
    print("✓ CALCULATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()