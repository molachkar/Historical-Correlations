"""
GOLD TRADING MASTER SYSTEM - FINAL VERSION
Fully compatible with Twelve Data free tier
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import joblib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATION
# ====================================================================

CONFIG = {
    'daily_model': 'C:\\Users\\PC\\Desktop\\MML\\lightgbm model\\lightgbm-direction-model.pkl',
    'daily_features': 'C:\\Users\\PC\\Desktop\\MML\\lightgbm model\\feature_names.pkl',
    'daily_threshold_path': 'C:\\Users\\PC\\Desktop\\MML\\lightgbm model\\best_threshold.pkl',
    'output_signal': 'C:\\Users\\PC\\Desktop\\MML\\master_signal.json',
    
    'twelve_data_key': '6195d22ee1754b65ac74dd4464f8d936',
    'newsapi_key': '07d638496f4741849ce55d2c42320107',
    'gnews_key': '3ca29aea34716d7b94e513c9463ef5f1',
    
    'risk_per_trade': 0.02,
    'account_size': 300,
    
    'cache_dir': 'C:\\Users\\PC\\Desktop\\MML\\cache',
    'cache_minutes': 60
}

# ====================================================================
# DATA FETCHER
# ====================================================================

def get_cached_or_fetch_twelve_data(symbol, interval, outputsize, cache_name):
    cache_dir = Path(CONFIG['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f"{cache_name}.pkl"
    
    if cache_file.exists():
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - cache_time < timedelta(minutes=CONFIG['cache_minutes']):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': CONFIG['twelve_data_key']
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if 'values' not in data:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            return None
        
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        return df
        
    except:
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

# ====================================================================
# LAYERS
# ====================================================================

class DailyModelLayer:
    def __init__(self):
        self.model = joblib.load(CONFIG['daily_model'])
        self.features = joblib.load(CONFIG['daily_features'])
        self.threshold = joblib.load(CONFIG['daily_threshold_path'])
    
    def get_signal(self, today_data):
        X = today_data[self.features].iloc[-1:].copy()
        X = X.apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0)
        
        proba = self.model.predict_proba(X)[0][1]
        
        if proba > self.threshold:
            signal = 'BUY'
            confidence = 'HIGH' if proba > 0.70 else 'MEDIUM' if proba > 0.60 else 'LOW'
        else:
            signal = 'HOLD'
            confidence = 'N/A'
        
        return {'signal': signal, 'probability': float(proba), 'confidence': confidence}

class EntryTimingLayer:
    def detect_entry_setup(self):
        df = get_cached_or_fetch_twelve_data('GLD', '1h', 168, 'gld_1h')
        
        if df is None or len(df) < 50:
            return {'enter': False, 'reason': 'No data', 'support': 0, 'resistance': 0, 'current_price': 0}
        
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        
        support = df['Low'].tail(48).min()
        resistance = df['High'].tail(48).max()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(current['Close'])
        
        dist_support = ((price - support) / support) * 100
        
        signals = []
        
        if dist_support < 1.5 and current['RSI'] < 50:
            signals.append({
                'type': 'PULLBACK', 'quality': 'HIGH',
                'entry': price, 'stop': support * 0.995, 'tp': price * 1.02
            })
        
        if price > resistance and current['Volume'] > current['Vol_MA'] * 1.3:
            signals.append({
                'type': 'BREAKOUT', 'quality': 'HIGH',
                'entry': price, 'stop': resistance, 'tp': price * 1.03
            })
        
        if prev['Close'] < prev['EMA_20'] and current['Close'] > current['EMA_20']:
            signals.append({
                'type': 'EMA_BOUNCE', 'quality': 'MEDIUM',
                'entry': price, 'stop': current['EMA_20'] * 0.995, 'tp': price * 1.02
            })
        
        if signals:
            best = max(signals, key=lambda x: 3 if x['quality'] == 'HIGH' else 2)
            return {
                'enter': True, 'signal_type': best['type'],
                'entry_price': best['entry'], 'stop_loss': best['stop'], 'take_profit': best['tp'],
                'support': float(support), 'resistance': float(resistance),
                'current_rsi': float(current['RSI'])
            }
        else:
            return {
                'enter': False, 'reason': 'No setup',
                'current_price': price, 'support': float(support), 'resistance': float(resistance),
                'current_rsi': float(current['RSI'])
            }

class NewsSentimentLayer:
    def get_news_sentiment(self):
        try:
            with open('C:\\Users\\PC\\Desktop\\MML\\news analyser\\gold_news_analysis.json') as f:
                data = json.load(f)
            bullish = data['sentiment_summary']['bullish']
            bearish = data['sentiment_summary']['bearish']
            total = bullish + bearish
            
            if total == 0:
                return {'sentiment': 'NEUTRAL', 'confidence': 0}
            
            pct = bullish / total
            sentiment = 'STRONG_BULLISH' if pct >= 0.65 else 'BULLISH' if pct >= 0.55 else 'NEUTRAL'
            return {'sentiment': sentiment, 'confidence': float(abs(pct - 0.5) * 2)}
        except:
            return {'sentiment': 'NEUTRAL', 'confidence': 0}

class VolatilityRegimeLayer:
    def get_volatility_regime(self):
        df = get_cached_or_fetch_twelve_data('GLD', '1day', 60, 'gld_1d')
        
        if df is None or len(df) < 14:
            return {'regime': 'NORMAL', 'position_multiplier': 1.0}
        
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        percentile = (atr.rank(pct=True).iloc[-1]) * 100
        
        if percentile > 80:
            return {'regime': 'HIGH', 'position_multiplier': 0.7, 'advice': 'High volatility, reduce 30%'}
        elif percentile < 20:
            return {'regime': 'LOW', 'position_multiplier': 0, 'advice': 'Low volatility, skip'}
        else:
            return {'regime': 'NORMAL', 'position_multiplier': 1.0, 'advice': 'Normal volatility'}

class CorrelationLayer:
    def get_correlations(self):
        gld = get_cached_or_fetch_twelve_data('GLD', '1day', 30, 'gld_corr')
        uup = get_cached_or_fetch_twelve_data('UUP', '1day', 30, 'uup_corr')
        vixy = get_cached_or_fetch_twelve_data('VIXY', '1day', 30, 'vixy_corr')  # Changed from VIX
        spy = get_cached_or_fetch_twelve_data('SPY', '1day', 30, 'spy_corr')
        
        if any(x is None or len(x) < 10 for x in [gld, uup, vixy, spy]):
            return {'regime_normal': True}
        
        df = pd.DataFrame({
            'gold': gld['Close'], 'dxy': uup['Close'],
            'vix': vixy['Close'], 'spy': spy['Close']
        }).dropna()
        
        if len(df) < 10:
            return {'regime_normal': True}
        
        corr_dxy = df['gold'].corr(df['dxy'])
        corr_vix = df['gold'].corr(df['vix'])
        
        regime_normal = corr_dxy < -0.3 or corr_vix > 0.2
        
        return {'regime_normal': regime_normal}

class MacroCalendarLayer:
    def get_todays_events(self):
        try:
            response = requests.get("https://www.investing.com/economic-calendar/", 
                                   headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            events = soup.find_all('i', class_='grayFullBullishIcon')
            return {'high_impact_today': len(events) > 0}
        except:
            return {'high_impact_today': False}

class VolumeSmartMoneyLayer:
    def get_volume_analysis(self):
        df = get_cached_or_fetch_twelve_data('GLD', '1h', 720, 'gld_vol')
        
        if df is None or len(df) < 50:
            return {'smart_money_signal': 'NEUTRAL'}
        
        df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
        
        obv_trend = df['OBV'].diff(5).iloc[-1]
        price_trend = df['Close'].diff(5).iloc[-1]
        
        if obv_trend > 0 and price_trend > 0:
            return {'smart_money_signal': 'ACCUMULATION'}
        elif obv_trend > 0 and price_trend < 0:
            return {'smart_money_signal': 'BULLISH_DIVERGENCE'}
        else:
            return {'smart_money_signal': 'NEUTRAL'}

# ====================================================================
# MASTER SYSTEM
# ====================================================================

class MasterTradingSystem:
    def __init__(self):
        self.daily_model = DailyModelLayer()
        self.entry = EntryTimingLayer()
        self.news = NewsSentimentLayer()
        self.vol = VolatilityRegimeLayer()
        self.corr = CorrelationLayer()
        self.macro = MacroCalendarLayer()
        self.volume = VolumeSmartMoneyLayer()
    
    def generate_master_signal(self, today_data):
        print("\n" + "="*70)
        print("🥇 GOLD MASTER TRADING SYSTEM")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        r = {}
        
        print("Layer 1: Daily Model...")
        r['daily'] = self.daily_model.get_signal(today_data)
        print(f"  {r['daily']['signal']} ({r['daily']['probability']:.1%})")
        
        if r['daily']['signal'] == 'HOLD':
            print("\n❌ NO TRADE\n")
            return self.save({'final': 'NO_TRADE', 'layers': r})
        
        print("\nLayer 2: Entry Timing...")
        r['entry'] = self.entry.detect_entry_setup()
        print(f"  Enter: {r['entry']['enter']}")
        print(f"  Support: ${r['entry']['support']:.2f}")
        print(f"  Resistance: ${r['entry']['resistance']:.2f}")
        print(f"  Current: ${r['entry'].get('current_price', r['entry'].get('entry_price', 0)):.2f}")
        
        print("\nLayer 3: News...")
        r['news'] = self.news.get_news_sentiment()
        print(f"  {r['news']['sentiment']}")
        
        print("\nLayer 4: Volatility...")
        r['vol'] = self.vol.get_volatility_regime()
        print(f"  {r['vol']['regime']}")
        
        print("\nLayer 5: Correlations...")
        r['corr'] = self.corr.get_correlations()
        print(f"  {'Normal' if r['corr']['regime_normal'] else 'Abnormal'}")
        
        print("\nLayer 6: Macro Events...")
        r['macro'] = self.macro.get_todays_events()
        print(f"  {'High impact today' if r['macro']['high_impact_today'] else 'None'}")
        
        print("\nLayer 7: Volume...")
        r['volume'] = self.volume.get_volume_analysis()
        print(f"  {r['volume']['smart_money_signal']}")
        
        # Scoring
        score = 0
        score += 3 if r['daily']['confidence'] == 'HIGH' else 2 if r['daily']['confidence'] == 'MEDIUM' else 1
        score += 1 if r['news']['sentiment'] in ['BULLISH', 'STRONG_BULLISH'] else 0
        score += 1 if r['vol']['regime'] == 'NORMAL' else -2 if r['vol']['regime'] == 'LOW' else 0
        score += 1 if r['corr']['regime_normal'] else 0
        score += 1 if r['volume']['smart_money_signal'] in ['ACCUMULATION', 'BULLISH_DIVERGENCE'] else 0
        score -= 2 if r['macro']['high_impact_today'] else 0
        
        print("\n" + "="*70)
        print(f"SCORE: {score}/7")
        print("="*70)
        
        can_enter = r['entry']['enter']
        
        if score >= 5 and can_enter:
            decision = 'ENTER_LONG_NOW'
            mult = r['vol']['position_multiplier']
            entry = r['entry']['entry_price']
            stop = r['entry']['stop_loss']
            tp = r['entry']['take_profit']
            risk = CONFIG['account_size'] * CONFIG['risk_per_trade'] * mult
            size = risk / (entry - stop) if (entry - stop) > 0 else 0
            
            print(f"\n✅ ENTER LONG NOW")
            print(f"   Entry: ${entry:.2f}")
            print(f"   Stop: ${stop:.2f}")
            print(f"   Target: ${tp:.2f}")
            print(f"   Size: {size:.2f} lots")
            print(f"   Risk: ${risk:.2f}\n")
            
        elif score >= 5:
            decision = 'WAIT_FOR_ENTRY'
            mult = r['vol']['position_multiplier']
            size = risk = entry = stop = tp = 0
            
            print(f"\n⏳ WAIT FOR ENTRY")
            print(f"   Watch for pullback to ${r['entry']['support']:.2f}")
            print(f"   Or breakout above ${r['entry']['resistance']:.2f}\n")
            
        elif score >= 3:
            decision = 'MONITOR_ONLY'
            mult = size = risk = entry = stop = tp = 0
            print(f"\n👀 MONITOR ONLY\n")
        else:
            decision = 'NO_TRADE'
            mult = size = risk = entry = stop = tp = 0
            print(f"\n❌ NO TRADE\n")
        
        return self.save({
            'final': decision, 'score': score,
            'entry_price': entry, 'stop_loss': stop, 'take_profit': tp,
            'position_size': size, 'risk': risk,
            'layers': r
        })
    
    def save(self, output):
        with open(CONFIG['output_signal'], 'w') as f:
            json.dump(output, f, indent=2, default=str)
        return output

# ====================================================================
# MAIN
# ====================================================================

def main():
    today_data = pd.read_csv("C:\\Users\\PC\\Desktop\\MML\\fetch.csv")
    
    for col in today_data.columns:
        today_data[col] = pd.to_numeric(today_data[col], errors='coerce')
    
    today_data = today_data.ffill().bfill().fillna(0)
    
    system = MasterTradingSystem()
    return system.generate_master_signal(today_data)

if __name__ == "__main__":
    main()