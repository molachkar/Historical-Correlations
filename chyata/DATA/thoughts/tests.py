import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import ta
import warnings
warnings.filterwarnings('ignore')

# Advanced imports for ML and sentiment
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not available. Install transformers, sklearn, torch for full functionality.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("key.env")

class Signal(Enum):
    """Enhanced trading signals"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(Enum):
    """Risk levels for position sizing"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class MarketData:
    """Comprehensive market data structure"""
    timestamp: datetime
    value: float
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: float
    macd: float
    macd_signal: float
    ema_50: float
    ema_200: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    adx: float
    stoch_k: float
    stoch_d: float

@dataclass
class TradingPosition:
    """Trading position management"""
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    risk_amount: float
    confidence: float

@dataclass
class RiskManagement:
    """Risk management parameters"""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_portfolio_risk: float = 0.06  # 6% max total portfolio risk
    max_correlation_exposure: float = 0.3  # 30% max correlated positions
    stop_loss_pct: float = 0.015  # 1.5% stop loss
    take_profit_pct: float = 0.045  # 4.5% take profit (3:1 R:R)

class EnhancedGoldTradingBot:
    """
    Advanced Real-Time XAU/USD Trading Bot
    Features: ML sentiment, technical analysis, risk management, real-time execution
    """
    
    def __init__(self):
        """Initialize enhanced trading bot"""
        self.risk_params = RiskManagement()
        self.positions: List[TradingPosition] = []
        self.initialize_components()
        self.setup_sentiment_analysis()
        
    def initialize_components(self):
        """Initialize all bot components"""
        self.configure_enhanced_metrics()
        self.setup_data_sources()
        self.initialize_ml_models()
        
    def configure_enhanced_metrics(self):
        """Configure enhanced metrics with new weights"""
        self.metrics_config = {
            # Fundamental Indicators
            'dxy': {'weight': 25, 'timeframe': '4h', 'threshold': 100, 'operator': 'lt'},
            'real_rates': {'weight': 20, 'timeframe': '4h', 'threshold': -0.5, 'operator': 'lt'},
            'vix': {'weight': 15, 'timeframe': '4h', 'threshold': 20, 'operator': 'gt'},
            'fed_policy': {'weight': 10, 'timeframe': 'daily', 'threshold': 0, 'operator': 'eq'},
            
            # Market Indicators
            'sp500': {'weight': 8, 'timeframe': '4h', 'threshold': 0, 'operator': 'lt'},
            'nasdaq': {'weight': 7, 'timeframe': '4h', 'threshold': 0, 'operator': 'lt'},
            'etf_flows': {'weight': 5, 'timeframe': 'daily', 'threshold': 20, 'operator': 'gt'},
            'cb_purchases': {'weight': 5, 'timeframe': 'weekly', 'threshold': 10, 'operator': 'gt'},
            
            # Sentiment & Alternative
            'sentiment_score': {'weight': 3, 'timeframe': '4h', 'threshold': 0.3, 'operator': 'gt'},
            'crypto_correlation': {'weight': 2, 'timeframe': '4h', 'threshold': -0.3, 'operator': 'lt'}
        }
        
        # Validate weights
        total_weight = sum(config['weight'] for config in self.metrics_config.values())
        if total_weight != 100:
            logger.warning(f"Metric weights sum to {total_weight}%, adjusting...")
            
    def setup_data_sources(self):
        """Setup all data source connections"""
        self.data_sources = {
            'yahoo': yf,
            'fred_api_key': os.getenv("FRED_API_KEY"),
            'news_api_key': os.getenv("NEWSAPI_KEY"),
            'alpha_vantage_key': os.getenv("ALPHA_VANTAGE_KEY")
        }
        
    def initialize_ml_models(self):
        """Initialize machine learning models"""
        if ML_AVAILABLE:
            try:
                # Fed sentiment analyzer
                self.fed_sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                
                # Price prediction model
                self.price_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                
                self.scaler = StandardScaler()
                logger.info("ML models initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ML models: {e}")
                self.fed_sentiment_model = None
                self.price_model = None
        else:
            self.fed_sentiment_model = None
            self.price_model = None
            
    def setup_sentiment_analysis(self):
        """Setup vector-based sentiment analysis"""
        self.sentiment_keywords = {
            'hawkish': ['hawkish', 'tightening', 'aggressive', 'raise rates', 'combat inflation'],
            'dovish': ['dovish', 'accommodative', 'supportive', 'lower rates', 'stimulus'],
            'uncertainty': ['uncertain', 'cautious', 'monitoring', 'data-dependent', 'flexible'],
            'crisis': ['crisis', 'emergency', 'urgent', 'unprecedented', 'severe']
        }
        
    def fetch_real_time_data(self) -> Dict[str, MarketData]:
        """Fetch all real-time market data"""
        logger.info("Fetching real-time market data...")
        data = {}
        
        # Fetch primary indicators
        data.update(self._fetch_forex_data())
        data.update(self._fetch_equity_data())
        data.update(self._fetch_volatility_data())
        data.update(self._fetch_economic_data())
        data.update(self._fetch_sentiment_data())
        
        return data
        
    def _fetch_forex_data(self) -> Dict[str, MarketData]:
        """Fetch forex and currency data"""
        data = {}
        
        try:
            # DXY (US Dollar Index)
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy.history(period="1d", interval="1h")
            if not dxy_data.empty:
                data['dxy'] = MarketData(
                    timestamp=datetime.now(),
                    value=float(dxy_data['Close'].iloc[-1]),
                    source="Yahoo Finance",
                    confidence=0.95
                )
            
            # XAU/USD for technical analysis
            gold = yf.Ticker("GC=F")
            gold_data = gold.history(period="5d", interval="1h")
            if not gold_data.empty:
                data['gold_price'] = MarketData(
                    timestamp=datetime.now(),
                    value=float(gold_data['Close'].iloc[-1]),
                    source="Yahoo Finance",
                    confidence=0.95,
                    metadata={'ohlc_data': gold_data}
                )
                
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            
        return data
        
    def _fetch_equity_data(self) -> Dict[str, MarketData]:
        """Fetch equity market data"""
        data = {}
        
        try:
            # S&P 500
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1d", interval="1h")
            if not sp500_data.empty:
                current_price = float(sp500_data['Close'].iloc[-1])
                prev_close = float(sp500_data['Close'].iloc[-2])
                pct_change = (current_price - prev_close) / prev_close * 100
                
                data['sp500'] = MarketData(
                    timestamp=datetime.now(),
                    value=pct_change,
                    source="Yahoo Finance",
                    confidence=0.95
                )
            
            # NASDAQ
            nasdaq = yf.Ticker("^IXIC")
            nasdaq_data = nasdaq.history(period="1d", interval="1h")
            if not nasdaq_data.empty:
                current_price = float(nasdaq_data['Close'].iloc[-1])
                prev_close = float(nasdaq_data['Close'].iloc[-2])
                pct_change = (current_price - prev_close) / prev_close * 100
                
                data['nasdaq'] = MarketData(
                    timestamp=datetime.now(),
                    value=pct_change,
                    source="Yahoo Finance",
                    confidence=0.95
                )
                
        except Exception as e:
            logger.error(f"Error fetching equity data: {e}")
            
        return data
        
    def _fetch_volatility_data(self) -> Dict[str, MarketData]:
        """Fetch volatility indicators"""
        data = {}
        
        try:
            # VIX
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d", interval="1h")
            if not vix_data.empty:
                data['vix'] = MarketData(
                    timestamp=datetime.now(),
                    value=float(vix_data['Close'].iloc[-1]),
                    source="Yahoo Finance",
                    confidence=0.95
                )
                
        except Exception as e:
            logger.error(f"Error fetching volatility data: {e}")
            
        return data
        
    def _fetch_economic_data(self) -> Dict[str, MarketData]:
        """Fetch economic indicators"""
        data = {}
        
        try:
            # Real interest rates (10Y TIPS)
            if self.data_sources['fred_api_key']:
                real_rates = self._fetch_fred_data('DFII10')
                if real_rates:
                    data['real_rates'] = real_rates
            
            # ETF flows (GLD tracking)
            gld = yf.Ticker("GLD")
            gld_data = gld.history(period="5d", interval="1h")
            if not gld_data.empty:
                volume_avg = gld_data['Volume'].tail(20).mean()
                current_volume = gld_data['Volume'].iloc[-1]
                volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
                
                data['etf_flows'] = MarketData(
                    timestamp=datetime.now(),
                    value=float(volume_ratio),
                    source="Yahoo Finance",
                    confidence=0.8
                )
                
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
            
        return data
        
    def _fetch_sentiment_data(self) -> Dict[str, MarketData]:
        """Fetch and analyze sentiment data"""
        data = {}
        
        try:
            # Vector-based sentiment analysis
            sentiment_score = self._analyze_fed_sentiment()
            data['sentiment_score'] = MarketData(
                timestamp=datetime.now(),
                value=sentiment_score,
                source="Vector Analysis",
                confidence=0.7
            )
            
            # Crypto correlation
            btc = yf.Ticker("BTC-USD")
            btc_data = btc.history(period="5d", interval="1h")
            
            if 'gold_price' in data and not btc_data.empty:
                gold_returns = data['gold_price'].metadata.get('ohlc_data', pd.DataFrame())
                if not gold_returns.empty:
                    # Calculate correlation
                    correlation = self._calculate_correlation(gold_returns, btc_data)
                    data['crypto_correlation'] = MarketData(
                        timestamp=datetime.now(),
                        value=correlation,
                        source="Correlation Analysis",
                        confidence=0.6
                    )
                    
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            
        return data
        
    def _fetch_fred_data(self, series_id: str) -> Optional[MarketData]:
        """Fetch FRED economic data"""
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.data_sources['fred_api_key'],
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' in data and data['observations']:
                obs = data['observations'][0]
                if obs['value'] != '.':
                    return MarketData(
                        timestamp=datetime.now(),
                        value=float(obs['value']),
                        source="FRED",
                        confidence=0.9
                    )
                    
        except Exception as e:
            logger.error(f"Error fetching FRED data: {e}")
            
        return None
        
    def _analyze_fed_sentiment(self) -> float:
        """Advanced Fed sentiment analysis using vector methods"""
        try:
            # Fetch recent Fed communications
            fed_texts = self._fetch_fed_communications()
            
            if not fed_texts:
                return 0.0
                
            # Vector-based sentiment scoring
            sentiment_scores = []
            
            for text in fed_texts:
                score = self._calculate_sentiment_vector(text)
                sentiment_scores.append(score)
                
            # Weight recent communications more heavily
            weights = np.linspace(0.5, 1.0, len(sentiment_scores))
            weighted_score = np.average(sentiment_scores, weights=weights)
            
            return float(weighted_score)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0
            
    def _calculate_sentiment_vector(self, text: str) -> float:
        """Calculate sentiment using vector-based approach"""
        text_lower = text.lower()
        
        # Score based on keyword presence and context
        hawkish_score = sum(1 for keyword in self.sentiment_keywords['hawkish'] 
                          if keyword in text_lower)
        dovish_score = sum(1 for keyword in self.sentiment_keywords['dovish'] 
                         if keyword in text_lower)
        uncertainty_score = sum(1 for keyword in self.sentiment_keywords['uncertainty'] 
                              if keyword in text_lower)
        crisis_score = sum(1 for keyword in self.sentiment_keywords['crisis'] 
                         if keyword in text_lower)
        
        # Calculate composite score (-1 to 1)
        total_keywords = hawkish_score + dovish_score + uncertainty_score + crisis_score
        
        if total_keywords == 0:
            return 0.0
            
        # Weighted sentiment calculation
        sentiment = (dovish_score * 0.5 + crisis_score * 0.3 - hawkish_score * 0.5 
                    - uncertainty_score * 0.2) / total_keywords
        
        return np.clip(sentiment, -1, 1)
        
    def _fetch_fed_communications(self) -> List[str]:
        """Fetch recent Fed communications"""
        # This would integrate with Fed's RSS feeds, press releases, etc.
        # For now, return mock data
        return [
            "The Federal Reserve remains committed to supporting the economy through appropriate monetary policy",
            "Recent economic indicators suggest a cautious approach to policy normalization",
            "Inflation expectations remain well-anchored, providing flexibility in policy decisions"
        ]
        
    def _calculate_correlation(self, series1: pd.DataFrame, series2: pd.DataFrame) -> float:
        """Calculate correlation between two price series"""
        try:
            # Align timeframes and calculate returns
            min_len = min(len(series1), len(series2))
            if min_len < 20:
                return 0.0
                
            returns1 = series1['Close'].tail(min_len).pct_change().dropna()
            returns2 = series2['Close'].tail(min_len).pct_change().dropna()
            
            if len(returns1) < 10 or len(returns2) < 10:
                return 0.0
                
            correlation = returns1.corr(returns2)
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
            
    def calculate_technical_indicators(self, gold_data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(gold_data['Close'], window=14).rsi().iloc[-1]
            
            # MACD
            macd_line = ta.trend.MACD(gold_data['Close']).macd().iloc[-1]
            macd_signal = ta.trend.MACD(gold_data['Close']).macd_signal().iloc[-1]
            
            # EMAs
            ema_50 = ta.trend.EMAIndicator(gold_data['Close'], window=50).ema_indicator().iloc[-1]
            ema_200 = ta.trend.EMAIndicator(gold_data['Close'], window=200).ema_indicator().iloc[-1]
            
            # Bollinger Bands
            bb_high = ta.volatility.BollingerBands(gold_data['Close']).bollinger_hband().iloc[-1]
            bb_low = ta.volatility.BollingerBands(gold_data['Close']).bollinger_lband().iloc[-1]
            bb_mid = ta.volatility.BollingerBands(gold_data['Close']).bollinger_mavg().iloc[-1]
            
            # ADX
            adx = ta.trend.ADXIndicator(gold_data['High'], gold_data['Low'], gold_data['Close']).adx().iloc[-1]
            
            # Stochastic
            stoch_k = ta.momentum.StochasticOscillator(gold_data['High'], gold_data['Low'], gold_data['Close']).stoch().iloc[-1]
            stoch_d = ta.momentum.StochasticOscillator(gold_data['High'], gold_data['Low'], gold_data['Close']).stoch_signal().iloc[-1]
            
            return TechnicalIndicators(
                rsi=float(rsi),
                macd=float(macd_line),
                macd_signal=float(macd_signal),
                ema_50=float(ema_50),
                ema_200=float(ema_200),
                bb_upper=float(bb_high),
                bb_lower=float(bb_low),
                bb_middle=float(bb_mid),
                adx=float(adx),
                stoch_k=float(stoch_k),
                stoch_d=float(stoch_d)
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
    def calculate_comprehensive_signal(self) -> Tuple[Signal, float, Dict[str, Any]]:
        """Calculate comprehensive trading signal"""
        logger.info("Calculating comprehensive trading signal...")
        
        # Fetch all data
        market_data = self.fetch_real_time_data()
        
        # Calculate fundamental score
        fundamental_score = self._calculate_fundamental_score(market_data)
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(market_data)
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment_score(market_data)
        
        # Combine scores with weights
        total_score = (
            fundamental_score * 0.6 +
            technical_score * 0.3 +
            sentiment_score * 0.1
        )
        
        # Determine signal
        signal = self._determine_signal(total_score)
        
        # Prepare analysis data
        analysis = {
            'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'sentiment_score': sentiment_score,
            'total_score': total_score,
            'market_data': market_data,
            'triggered_metrics': self._get_triggered_metrics(market_data)
        }
        
        return signal, total_score, analysis
        
    def _calculate_fundamental_score(self, market_data: Dict[str, MarketData]) -> float:
        """Calculate fundamental analysis score"""
        score = 0.0
        total_weight = 0
        
        for metric, config in self.metrics_config.items():
            if metric in market_data:
                data = market_data[metric]
                if self._is_metric_triggered(metric, data.value, config):
                    score += config['weight']
                total_weight += config['weight']
                
        return (score / total_weight * 100) if total_weight > 0 else 0.0
        
    def _calculate_technical_score(self, market_data: Dict[str, MarketData]) -> float:
        """Calculate technical analysis score"""
        if 'gold_price' not in market_data:
            return 50.0  # Neutral
            
        gold_data = market_data['gold_price'].metadata.get('ohlc_data')
        if gold_data is None or gold_data.empty:
            return 50.0
            
        try:
            tech_indicators = self.calculate_technical_indicators(gold_data)
            
            # Score technical indicators
            scores = []
            
            # RSI scoring
            if tech_indicators.rsi < 30:
                scores.append(80)  # Oversold - bullish
            elif tech_indicators.rsi > 70:
                scores.append(20)  # Overbought - bearish
            else:
                scores.append(50)  # Neutral
                
            # MACD scoring
            if tech_indicators.macd > tech_indicators.macd_signal:
                scores.append(70)  # Bullish crossover
            else:
                scores.append(30)  # Bearish crossover
                
            # EMA scoring
            current_price = float(gold_data['Close'].iloc[-1])
            if current_price > tech_indicators.ema_50 > tech_indicators.ema_200:
                scores.append(80)  # Strong uptrend
            elif current_price < tech_indicators.ema_50 < tech_indicators.ema_200:
                scores.append(20)  # Strong downtrend
            else:
                scores.append(50)  # Mixed signals
                
            # Bollinger Bands scoring
            if current_price < tech_indicators.bb_lower:
                scores.append(75)  # Oversold
            elif current_price > tech_indicators.bb_upper:
                scores.append(25)  # Overbought
            else:
                scores.append(50)  # Within bands
                
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 50.0
            
    def _calculate_sentiment_score(self, market_data: Dict[str, MarketData]) -> float:
        """Calculate sentiment-based score"""
        if 'sentiment_score' not in market_data:
            return 50.0
            
        sentiment_value = market_data['sentiment_score'].value
        
        # Convert sentiment (-1 to 1) to score (0 to 100)
        score = (sentiment_value + 1) * 50
        
        return np.clip(score, 0, 100)
        
    def _determine_signal(self, total_score: float) -> Signal:
        """Determine trading signal based on total score"""
        if total_score >= 80:
            return Signal.STRONG_BUY
        elif total_score >= 65:
            return Signal.BUY
        elif total_score >= 55:
            return Signal.WEAK_BUY
        elif total_score >= 45:
            return Signal.HOLD
        elif total_score >= 35:
            return Signal.WEAK_SELL
        elif total_score >= 20:
            return Signal.SELL
        else:
            return Signal.STRONG_SELL
            
    def _is_metric_triggered(self, metric: str, value: float, config: Dict) -> bool:
        """Check if metric is triggered based on conditions"""
        if value is None:
            return False
            
        operator = config['operator']
        threshold = config['threshold']
        
        if operator == 'gt':
            return value > threshold
        elif operator == 'lt':
            return value < threshold
        elif operator == 'eq':
            return abs(value - threshold) < 0.01
        else:
            return False
            
    def _get_triggered_metrics(self, market_data: Dict[str, MarketData]) -> List[str]:
        """Get list of triggered metrics"""
        triggered = []
        
        for metric, config in self.metrics_config.items():
            if metric in market_data:
                data = market_data[metric]
                if self._is_metric_triggered(metric, data.value, config):
                    triggered.append(metric)
                    
        return triggered
        
    def create_entry_exit_plan(self, signal: Signal, current_price: float, 
                             analysis: Dict) -> Dict[str, Any]:
        """Create comprehensive entry and exit plan"""
        plan = {
            'signal': signal.value,
            'current_price': current_price,
            'confidence': analysis['total_score'],
            'entry_plan': {},
            'exit_plan': {},
            'risk_management': {}
        }
        
        if signal in [Signal.STRONG_BUY, Signal.BUY, Signal.WEAK_BUY]:
            # Long position plan
            plan['entry_plan'] = {
                'direction': 'LONG',
                'entry_price': current_price,
                'position_size': self._calculate_position_size(signal, analysis),
                'entry_conditions': 'Market order on signal confirmation'
            }
            
            plan['exit_plan'] = {
                'stop_loss': current_price * (1 - self.risk_params.stop_loss_pct),
                'take_profit': current_price * (1 + self.risk_params.take_profit_pct),
                'trailing_stop': True,
                'partial_profit_levels': [
                    current_price * 1.015,  # 1.5% partial profit
                    current_price * 1.030   # 3.0% partial profit
                ]
            }
            
        elif signal in [Signal.STRONG_SELL, Signal.SELL, Signal.WEAK_SELL]:
            # Short position plan
            plan['entry_plan'] = {
                'direction': 'SHORT',
                'entry_price': current_price,
                'position_size': self._calculate_position_size(signal, analysis),
                'entry_conditions': 'Market order on signal confirmation'
            }
            
            plan['exit_plan'] = {
                'stop_loss': current_price * (1 + self.risk_params.stop_loss_pct),
                'take_profit': current_price * (1 - self.risk_params.take_profit_pct),
                'trailing_stop': True,
                'partial_profit_levels': [
                    current_price * 0.985,  # 1.5% partial profit
                    current_price * 0.970   # 3.0% partial profit
                ]
            }
            
        else:
            # Hold position
            plan['entry_plan'] = {
                'direction': 'HOLD',
                'recommendation': 'Wait for clearer signal'
            }
            
        # Risk management details
        plan['risk_management'] = {
            'max_risk_per_trade': f"{self.risk_params.max_risk_per_trade * 100:.1f}%",
            'risk_reward_ratio': f"1:{self.risk_params.take_profit_pct / self.risk_params.stop_loss_pct:.1f}",
            'position_size_method': 'Kelly Criterion adjusted',
            'correlation_check': 'Monitor USD correlated positions'
        }
        
        return plan
        
    def _calculate_position_size(self, signal: Signal, analysis: Dict) -> float:
        """Calculate optimal position size based on Kelly Criterion"""
        base_size = 0.01  # 1% base position
        
        # Adjust based on signal strength
        signal_multiplier = {
            Signal.STRONG_BUY: 1.5,
            Signal.BUY: 1.2,
            Signal.WEAK_BUY: 0.8,
            Signal.HOLD: 0.0,
            Signal.WEAK_SELL: 0.8,
            Signal.SELL: 1.2,
            Signal.