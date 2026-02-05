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
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        return tenkan_sen, kijun_sen, senkou_a, senkou_span_b
    
    # ==================== ADVANCED INDICATORS ====================
    
    def calculate_parabolic_sar(self, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        length = len(self.df)
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        
        psar = np.zeros(length)
        bull = True
        af = af_start
        ep = low[0]
        hp = high[0]
        lp = low[0]
        
        psar[0] = close[0]
        
        for i in range(1, length):
            if bull:
                psar[i] = psar[i-1] + af * (hp - psar[i-1])
                if low[i] < psar[i]:
                    bull = False
                    psar[i] = hp
                    lp = low[i]
                    af = af_start
            else:
                psar[i] = psar[i-1] + af * (lp - psar[i-1])
                if high[i] > psar[i]:
                    bull = True
                    psar[i] = lp
                    hp = high[i]
                    af = af_start
            
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_increment, af_max)
                if low[i-1] < psar[i]:
                    psar[i] = low[i-1]
                if low[i-2] < psar[i]:
                    psar[i] = low[i-2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_increment, af_max)
                if high[i-1] > psar[i]:
                    psar[i] = high[i-1]
                if high[i-2] > psar[i]:
                    psar[i] = high[i-2]
        
        return pd.Series(psar, index=self.df.index)
    
    def calculate_awesome_oscillator(self, fast=5, slow=34):
        """Calculate Awesome Oscillator"""
        median_price = (self.df['High'] + self.df['Low']) / 2
        ao = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
        return ao
    
    def calculate_williams_r(self, period=14):
        """Calculate Williams %R"""
        highest_high = self.df['High'].rolling(window=period).max()
        lowest_low = self.df['Low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - self.df['Close']) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_cci(self, period=20):
        """Calculate Commodity Channel Index"""
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci
    
    def calculate_mfi(self, period=14):
        """Calculate Money Flow Index (requires Volume)"""
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        mf = tp * self.df['Volume']
        
        mf_pos = pd.Series(0.0, index=self.df.index)
        mf_neg = pd.Series(0.0, index=self.df.index)
        
        for i in range(1, len(self.df)):
            if tp.iloc[i] > tp.iloc[i-1]:
                mf_pos.iloc[i] = mf.iloc[i]
            elif tp.iloc[i] < tp.iloc[i-1]:
                mf_neg.iloc[i] = mf.iloc[i]
        
        mf_pos_sum = mf_pos.rolling(window=period).sum()
        mf_neg_sum = mf_neg.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (mf_pos_sum / mf_neg_sum)))
        return mfi
    
    def calculate_roc(self, period=12):
        """Calculate Rate of Change"""
        roc = ((self.df['Close'] - self.df['Close'].shift(period)) / 
               self.df['Close'].shift(period)) * 100
        return roc
    
    def calculate_hurst_exponent(self, window=100):
        """Calculate Hurst Exponent (mean reversion indicator)"""
        def hurst(ts):
            if len(ts) < 20:
                return np.nan
            
            lags = range(2, min(20, len(ts)//2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            
            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return np.nan
        
        hurst_values = self.df['Close'].rolling(window=window).apply(hurst, raw=False)
        return hurst_values
    
    def calculate_vwap(self):
        """Calculate Volume Weighted Average Price (simple daily VWAP)"""
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        vwap = (tp * self.df['Volume']).cumsum() / self.df['Volume'].cumsum()
        return vwap
    
    # ==================== EFFICIENCY RATIO ====================
    
    def calculate_efficiency_ratio(self, period=10):
        """
        Calculate Kaufman's Efficiency Ratio
        Measures price movement efficiency
        """
        change = abs(self.df['Close'] - self.df['Close'].shift(period))
        volatility = abs(self.df['Close'].diff()).rolling(window=period).sum()
        er = change / volatility
        return er
    
    # ==================== MASTER CALCULATION ====================
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        print("Calculating all technical indicators...")
        
        # EMAs
        print("  ✓ Calculating EMAs...")
        self.calculate_all_emas()
        
        # Volatility
        print("  ✓ Calculating Volatility indicators...")
        self.df['Parkinson_Vol'] = self.calculate_parkinson_volatility()
        self.df['YangZhang_Vol'] = self.calculate_yang_zhang_volatility()
        self.df['Vol_Percentile'] = self.calculate_volatility_percentile(self.df['Parkinson_Vol'])
        self.df['Vol_Regime'] = self.classify_volatility_regime(self.df['Vol_Percentile'])
        
        # Momentum
        print("  ✓ Calculating Momentum indicators...")
        self.df['RSI'] = self.calculate_rsi()
        self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = self.calculate_macd()
        
        # Trend
        print("  ✓ Calculating Trend indicators...")
        self.df['ADX'], self.df['Plus_DI'], self.df['Minus_DI'] = self.calculate_adx()
        
        # Bollinger Bands
        print("  ✓ Calculating Bollinger Bands...")
        self.df['BB_Upper'], self.df['BB_Middle'], self.df['BB_Lower'], self.df['BB_Width'] = \
            self.calculate_bollinger_bands()
        
        # ATR
        print("  ✓ Calculating ATR...")
        self.df['ATR'] = self.calculate_atr()
        
        # Stochastic
        print("  ✓ Calculating Stochastic...")
        self.df['Stoch_K'], self.df['Stoch_D'] = self.calculate_stochastic()
        
        # Ichimoku
        print("  ✓ Calculating Ichimoku Cloud...")
        self.df['Ichimoku_TK'], self.df['Ichimoku_KJ'], self.df['Ichimoku_SA'], self.df['Ichimoku_SB'] = \
            self.calculate_ichimoku()
        
        # Advanced
        print("  ✓ Calculating Advanced indicators...")
        self.df['PSAR'] = self.calculate_parabolic_sar()
        self.df['AO'] = self.calculate_awesome_oscillator()
        self.df['Williams_R'] = self.calculate_williams_r()
        self.df['CCI'] = self.calculate_cci()
        self.df['MFI'] = self.calculate_mfi()
        self.df['ROC'] = self.calculate_roc()
        self.df['Hurst'] = self.calculate_hurst_exponent()
        self.df['VWAP'] = self.calculate_vwap()
        self.df['Efficiency_Ratio'] = self.calculate_efficiency_ratio()
        
        print("✓ All indicators calculated successfully!")
        
        return self.df
    
    # ==================== OUTPUT FORMATTING ====================
    
    def format_to_deepin_structure(self, start_row=200):
        """
        Format output to match deepin_daily.json structure
        Start from row 200 to ensure all indicators have valid values
        """
        results = {}
        
        for idx in range(start_row, len(self.df)):
            row = self.df.iloc[idx]
            date_str = row['Date'].strftime('%Y-%m-%d')
            
            # Hurst state classification
            hurst_val = row['Hurst']
            if pd.notna(hurst_val):
                if hurst_val < 0.5:
                    hurst_state = "mean_reverting"
                elif hurst_val > 0.5:
                    hurst_state = "trending"
                else:
                    hurst_state = "random_walk"
            else:
                hurst_state = "unknown"
            
            results[date_str] = {
                "XAUUSD": {
                    "name": "Gold",
                    "price": {
                        "o": round(row['Open'], 2),
                        "h": round(row['High'], 2),
                        "l": round(row['Low'], 2),
                        "c": round(row['Close'], 2),
                        "v": int(row['Volume'])
                    },
                    "ema": {
                        "e9": round(row['EMA_9'], 2),
                        "e21": round(row['EMA_21'], 2),
                        "e50": round(row['EMA_50'], 2),
                        "e200": round(row['EMA_200'], 2)
                    },
                    "volatility": {
                        "parkinson": round(row['Parkinson_Vol'], 4),
                        "yang_zhang": round(row['YangZhang_Vol'], 4),
                        "regime": row['Vol_Regime'],
                        "percentile": round(row['Vol_Percentile'], 2)
                    },
                    "hurst": {
                        "value": round(row['Hurst'], 4) if pd.notna(row['Hurst']) else None,
                        "state": hurst_state
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
            }
        
        return results


def main():
    """Main execution function"""
    
    print("="*70)
    print("TECHNICAL INDICATORS CALCULATOR")
    print("Matches deepin_daily.json structure")
    print("="*70)
    print()
    
    # Set paths relative to script location
    # Script is in NEW directory, data is in NEW/DATA/XAUUSD.csv
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
    
    # Calculate indicators
    calc = TechnicalIndicators(df)
    result_df = calc.calculate_all_indicators()
    
    print()
    print("="*70)
    print("FORMATTING OUTPUT")
    print("="*70)
    print()
    
    # Format to JSON structure (skip first 200 rows for indicator warmup)
    print("Formatting to deepin_daily.json structure...")
    print("  (Starting from row 200 to ensure all indicators are valid)")
    
    json_output = calc.format_to_deepin_structure(start_row=200)
    
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
    
    # Save CSV with all indicators
    output_csv = os.path.join(output_dir, "xauusd_all_indicators.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"✓ Full CSV saved to: {output_csv}")
    
    # Summary statistics
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total rows processed: {len(result_df)}")
    print(f"Rows in JSON output: {len(json_output)} (starting from row 200)")
    print(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")
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
    print("  ✓ Hurst Exponent")
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