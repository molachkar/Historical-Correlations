# gold_signal_v5_fixed.py
"""
Gold + Market Sentiment System - Fixed & focused
- No crypto fetching (only equities/indices, inflation/macro, gold price)
- Robust caching (timestamp serialization fixed)
- Simple technical layer
- Regular + Breaking news (with freshness and de-dup)
- Financial-analytics oriented LLM prompts (GROQ)
- Environment variables for API keys
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ta
import feedparser

# ---------------------
# Logging
# ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("gold_signal_v5")

# ---------------------
# Config
# ---------------------
class Config:
    # Read API keys from environment - DO NOT hardcode
    FRED_API_KEY = os.getenv("FRED_API_KEY", "a7a7e3fcfe30edeed9edc26dbea210b7")
    TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "6195d22ee1754b65ac74dd4464f8d936")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "07d638496f4741849ce55d2c42320107")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    BASE_DIR = Path(os.getenv("GOLD_BASE_DIR", Path.home() / "gold_signals"))
    CACHE_DIR = BASE_DIR / "cache"
    OUTPUT_FILE = BASE_DIR / "gold_signal_v5_output.json"
    VECTOR_CACHE = CACHE_DIR / "fixed_vectors.npz"
    HEADLINE_CACHE = CACHE_DIR / "seen_headlines.json"
    DATA_CACHE = CACHE_DIR / "data_cache.json"

    # Tunables
    FETCH_DELAY = float(os.getenv("FETCH_DELAY", 0.2))
    NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", 168))  # 7 days
    BREAKING_HOURS = int(os.getenv("BREAKING_HOURS", 3))
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 72))

    # ensure dirs
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------
# Fixed Descriptions & Vectors
# ---------------------
MACRO_DESC = (
    "Monetary policy, interest rates, inflation, employment, GDP, and market risk indicators "
    "affecting equities, bonds and currency movements."
)

GOLD_DESC = (
    "Gold price drivers: real yields, inflation expectations, dollar strength, central bank buying, and safe-haven demand."
)

class VectorManager:
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    @staticmethod
    def load_or_create_vectors():
        if Config.VECTOR_CACHE.exists():
            try:
                logger.info("✓ Loading cached vectors...")
                data = np.load(str(Config.VECTOR_CACHE))
                return data['macro'], data['gold']
            except Exception as e:
                logger.warning("Failed loading vector cache, recreating: %s", e)

        logger.info("Creating fixed vectors (one-time)...")
        model = VectorManager.model()
        macro_vec = model.encode(MACRO_DESC, show_progress_bar=False)
        gold_vec = model.encode(GOLD_DESC, show_progress_bar=False)
        np.savez_compressed(str(Config.VECTOR_CACHE), macro=macro_vec, gold=gold_vec)
        logger.info("✓ Vectors cached for future runs")
        return macro_vec, gold_vec

MACRO_VEC, GOLD_VEC = VectorManager.load_or_create_vectors()
EMBED_MODEL = VectorManager.model()

# ---------------------
# Utilities
# ---------------------
def now_utc():
    return datetime.now(timezone.utc)

def hours_between(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds()) / 3600.0

def safe_json_dump(path, obj):
    """Dump JSON with fallback for non-serializable types (convert to ISO strings)."""
    def default(o):
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, default=default, indent=2)

def safe_json_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------------
# DataFetcher (FRED, Gold price via TwelveData, Fear&Greed)
# ---------------------
class DataFetcher:
    def __init__(self):
        # Initialize cache file if needed
        if not Config.DATA_CACHE.exists():
            safe_json_dump(Config.DATA_CACHE, {})

    def _cache(self, key, data):
        try:
            cache = safe_json_load(Config.DATA_CACHE)
        except Exception:
            cache = {}
        cache[key] = {'data': data, 'time': now_utc().isoformat()}
        safe_json_dump(Config.DATA_CACHE, cache)

    def _load(self, key, max_hours=None):
        try:
            cache = safe_json_load(Config.DATA_CACHE)
        except Exception as e:
            logger.debug("Cache read failed: %s", e)
            return None
        if key not in cache:
            return None
        ts = date_parser.parse(cache[key]['time'])
        max_hours = Config.CACHE_TTL_HOURS if max_hours is None else max_hours
        age_h = hours_between(now_utc(), ts)
        if age_h > max_hours:
            logger.info("Cache %s expired (%.2f h > %s h)", key, age_h, max_hours)
            return None
        return cache[key]['data']

    def get_fred_latest(self, series_id):
        """Get latest value for a FRED series (single latest observation)."""
        cache_key = f"fred_{series_id}"
        cached = self._load(cache_key, max_hours=12)
        if cached is not None:
            return cached
        if not Config.FRED_API_KEY:
            logger.warning("FRED_API_KEY missing")
            return None
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={'series_id': series_id, 'api_key': Config.FRED_API_KEY, 'file_type': 'json', 'sort_order': 'desc', 'limit': 1},
                timeout=10
            )
            r.raise_for_status()
            j = r.json()
            obs = j.get('observations', [])
            if not obs:
                return None
            val = obs[0].get('value')
            if val in (None, '.', ''):
                return None
            value = float(val)
            self._cache(cache_key, value)
            return value
        except Exception as e:
            logger.exception("get_fred_latest error: %s", e)
            return self._load(cache_key, max_hours=72)

    def get_gold_4h(self):
        """Fetch gold 4H candles (TwelveData). Cache serialized safely (ISO strings)."""
        cache_key = "gold_4h"
        cached = self._load(cache_key, max_hours=1)
        if cached is not None:
            # cached is a list of dicts - convert to DataFrame and reparse datetimes
            df = pd.DataFrame(cached)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            return df

        if not Config.TWELVE_DATA_KEY:
            logger.warning("TWELVE_DATA_KEY missing")
            return None

        try:
            r = requests.get("https://api.twelvedata.com/time_series", params={
                'symbol': 'XAU/USD', 'interval': '4h', 'apikey': Config.TWELVE_DATA_KEY, 'outputsize': 42, 'format': 'JSON'
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            if 'values' not in data:
                logger.warning("TwelveData returned unexpected payload")
                return self._load(cache_key, max_hours=72)
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            for c in ['open', 'high', 'low', 'close']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.sort_values('datetime').reset_index(drop=True)
            # convert datetimes to ISO strings for JSON serializable cache
            temp = df.copy()
            if 'datetime' in temp.columns:
                temp['datetime'] = temp['datetime'].dt.tz_localize(None).astype(str)
            self._cache(cache_key, temp.to_dict('records'))
            time.sleep(Config.FETCH_DELAY)
            return df
        except Exception as e:
            logger.exception("get_gold_4h failed: %s", e)
            return self._load(cache_key, max_hours=72)

    def get_fear_and_greed(self):
        cache_key = 'fear_greed'
        cached = self._load(cache_key, max_hours=6)
        if cached:
            return cached
        try:
            r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", timeout=10)
            r.raise_for_status()
            j = r.json()
            data = j.get('fear_and_greed', {})
            score = data.get('score')
            rating = data.get('rating')
            if score is None:
                return None
            result = {'value': int(score), 'rating': rating, 'normalized': (int(score) - 50) / 50.0}
            self._cache(cache_key, result)
            return result
        except Exception as e:
            logger.exception("get_fear_and_greed failed: %s", e)
            return self._load(cache_key, max_hours=24)

# ---------------------
# NewsFetcher (NewsAPI + RSS), with seen-headlines dedup + breaking/regular separation
# ---------------------
class NewsFetcher:
    def __init__(self):
        if Config.HEADLINE_CACHE.exists():
            try:
                self.seen = safe_json_load(Config.HEADLINE_CACHE)
            except Exception:
                self.seen = {}
        else:
            self.seen = {}

    def _save_seen(self):
        safe_json_dump(Config.HEADLINE_CACHE, self.seen)

    def _hash(self, text):
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _is_seen(self, text):
        h = self._hash(text)
        if h in self.seen:
            try:
                ts = date_parser.parse(self.seen[h])
                return hours_between(now_utc(), ts) < 12
            except Exception:
                return False
        return False

    def _mark_seen(self, text):
        self.seen[self._hash(text)] = now_utc().isoformat()
        self._save_seen()

    def fetch_newsapi(self, query, lookback_hours=Config.NEWS_LOOKBACK_HOURS):
        if not Config.NEWSAPI_KEY:
            logger.warning("NEWSAPI_KEY missing")
            return []
        try:
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': Config.NEWSAPI_KEY,
                'from': (now_utc() - timedelta(hours=lookback_hours)).isoformat() + 'Z',
                'pageSize': 100
            }
            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            r.raise_for_status()
            articles = r.json().get('articles', [])
            out = []
            for a in articles:
                title = ' '.join((a.get('title') or '').split())
                if len(title) < 10 or self._is_seen(title):
                    continue
                pub = a.get('publishedAt')
                try:
                    pub_dt = date_parser.parse(pub) if pub else None
                    age_h = hours_between(now_utc(), pub_dt) if pub_dt else 999
                except Exception:
                    age_h = 999
                category = 'breaking' if age_h < Config.BREAKING_HOURS else 'recent' if age_h < 24 else 'background'
                out.append({'title': title, 'source': (a.get('source') or {}).get('name', ''), 'age_hours': round(age_h,1), 'category': category})
                self._mark_seen(title)
            time.sleep(Config.FETCH_DELAY)
            return out
        except Exception as e:
            logger.exception("fetch_newsapi error: %s", e)
            return []

    def fetch_rss(self, url, source_name, lookback_hours=72):
        try:
            feed = feedparser.parse(url)
            out = []
            for entry in (feed.entries or [])[:80]:
                title = ' '.join((entry.get('title') or '').split())
                if len(title) < 10 or self._is_seen(title):
                    continue
                pub_dt = None
                if 'published' in entry:
                    try:
                        pub_dt = date_parser.parse(entry.published)
                    except Exception:
                        pub_dt = None
                if not pub_dt and 'updated' in entry:
                    try:
                        pub_dt = date_parser.parse(entry.updated)
                    except Exception:
                        pub_dt = None
                age_h = hours_between(now_utc(), pub_dt) if pub_dt else 999
                if age_h > lookback_hours:
                    continue
                cat = 'breaking' if age_h < Config.BREAKING_HOURS else 'recent'
                out.append({'title': title, 'source': source_name, 'age_hours': round(age_h,1), 'category': cat})
                self._mark_seen(title)
            time.sleep(Config.FETCH_DELAY)
            return out
        except Exception as e:
            logger.exception("fetch_rss failed: %s", e)
            return []

    def fetch_all(self):
        logger.info("FETCHING NEWS (ALL SOURCES)")
        all_headlines = []

        # Use financial-relevant query - not crypto
        query = '("Federal Reserve" OR "interest rates" OR "inflation" OR "CPI" OR "PCE" OR "unemployment" OR "GDP" OR "S&P 500" OR "equities" OR gold OR XAUUSD)'

        logger.info("NewsAPI...")
        all_headlines.extend(self.fetch_newsapi(query, lookback_hours=Config.NEWS_LOOKBACK_HOURS))

        rss_sources = [
            ('https://www.reuters.com/rssFeed/businessNews', 'Reuters', 72),
            ('https://www.bloomberg.com/feed/podcast/etf.xml', 'Bloomberg', 72),  # example - RSS endpoints vary
            ('https://www.kitco.com/rss/KitcoNews-GoldSilver.xml', 'Kitco', 48),
            ('https://feeds.marketwatch.com/marketwatch/topstories/', 'MarketWatch', 48),
            ('https://news.google.com/rss/search?q=Federal+Reserve+inflation&hl=en', 'Google News', 24),
        ]
        for url, name, hrs in rss_sources:
            logger.info("RSS: %s", name)
            all_headlines.extend(self.fetch_rss(url, name, lookback_hours=hrs))

        logger.info("Total headlines collected: %d", len(all_headlines))
        breaking = [h for h in all_headlines if h['category'] == 'breaking']
        regular = [h for h in all_headlines if h['category'] != 'breaking']
        logger.info("Breaking: %d | Regular: %d", len(breaking), len(regular))
        return {'breaking': breaking, 'regular': regular}

# ---------------------
# Headline semantic filter
# ---------------------
def filter_headlines(headlines, threshold=0.28):
    macro_hits = []
    gold_hits = []
    if not headlines:
        return macro_hits, gold_hits
    for h in headlines:
        try:
            v = EMBED_MODEL.encode(h['title'], show_progress_bar=False)
            macro_sim = float(cosine_similarity([MACRO_VEC], [v])[0][0])
            gold_sim = float(cosine_similarity([GOLD_VEC], [v])[0][0])
            if macro_sim > threshold:
                h['relevance'] = round(macro_sim, 4)
                macro_hits.append(h)
            if gold_sim > threshold + 0.02:
                h['relevance'] = round(gold_sim, 4)
                gold_hits.append(h)
        except Exception as e:
            logger.debug("Embedding error: %s", e)
    return macro_hits, gold_hits

# ---------------------
# Technical analysis (simple)
# ---------------------
def calculate_technicals(df):
    """
    Minimal technical layer:
    - price, ema20, rsi(14), atr(14), support/resistance (28 candles)
    """
    if df is None or len(df) < 20:
        return None
    dfc = df.copy().reset_index(drop=True)
    try:
        close = dfc['close'].astype(float)
        high = dfc['high'].astype(float)
        low = dfc['low'].astype(float)

        ema20 = close.ewm(span=20).mean().iloc[-1]
        rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]

        recent = dfc.tail(28)
        resistance = float(recent['high'].max())
        support = float(recent['low'].min())
        resistance_2 = float(recent['high'].nlargest(3).iloc[-1]) if len(recent) >= 3 else resistance
        support_2 = float(recent['low'].nsmallest(3).iloc[-1]) if len(recent) >= 3 else support
        current = float(close.iloc[-1])
        trend = 'Uptrend' if current > ema20 else 'Downtrend'

        return {
            'price': round(current, 2),
            'ema20': round(float(ema20), 2),
            'rsi': round(float(rsi), 2),
            'atr': round(float(atr), 2),
            'resistance': round(resistance, 2),
            'support': round(support, 2),
            'resistance_2': round(resistance_2, 2),
            'support_2': round(support_2, 2),
            'trend': trend
        }
    except Exception as e:
        logger.exception("calculate_technicals failed: %s", e)
        return None

# ---------------------
# GROQ LLM wrapper + finance prompts
# ---------------------
def groq_call(prompt: str, temperature=0.2, max_tokens=800):
    if not Config.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not configured - skipping LLM")
        return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        r.raise_for_status()
        j = r.json()
        choices = j.get('choices') or []
        if not choices:
            return None
        return choices[0].get('message', {}).get('content')
    except Exception as e:
        logger.exception("groq_call failed: %s", e)
        return None

def parse_json_block(text):
    if not text:
        return {}
    try:
        s = text.strip()
        # Extract JSON block if wrapped
        if '```json' in s:
            s = s.split('```json',1)[1].split('```',1)[0]
        elif '```' in s:
            s = s.split('```',1)[1].split('```',1)[0]
        # find first {...} pair
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except Exception:
        # best-effort fallback: return empty
        logger.debug("parse_json_block failed to parse LLM output")
        return {}

def build_regular_prompt(indices, fear_greed, macro_headlines, gold_headlines, tech):
    """
    Structured finance analysis prompt:
    - Ask for market sentiment (risk_on/risk_off), numeric scores -1..1
    - Ask for gold sentiment + score -1..1
    - Provide 2-line summary and key drivers
    """
    idx_str = f"S&P: {indices.get('sp500')}, VIX: {indices.get('vix')}, Fed funds (DFF): {indices.get('fed_rate')}, 10y yield: {indices.get('yield_10y')}"
    fg_str = f"Fear&Greed: {fear_greed.get('value') if fear_greed else 'N/A'}"
    macro_list = "\n".join([f"- {h['title']}" for h in (macro_headlines[:20] or [])])
    gold_list = "\n".join([f"- {h['title']}" for h in (gold_headlines[:15] or [])])
    tech_str = f"Gold Technical: price ${tech['price'] if tech else 'N/A'}, rsi {tech['rsi'] if tech else 'N/A'}, trend {tech['trend'] if tech else 'N/A'}, support ${tech['support'] if tech else 'N/A'}, resistance ${tech['resistance'] if tech else 'N/A'}"

    prompt = f"""
You are a senior financial analyst. Given the market data and headlines below, produce a concise JSON object only (no extra text) with the following fields:
- market_sentiment: one of ["risk_on","risk_off"]
- market_score: float between -1.0 (strong risk_off) and 1.0 (strong risk_on)
- gold_sentiment: one of ["BULLISH","BEARISH","NEUTRAL"]
- gold_score: float between -1.0 (very bearish) and 1.0 (very bullish)
- summary: two-sentence human summary (max 30 words total)
- drivers: array of up to 4 bullet drivers (short phrases)

Market indices:
{idx_str}
{fg_str}

Macro headlines (recent):
{macro_list}

Gold headlines (recent / relevant):
{gold_list}

{tech_str}

Return strictly valid JSON. Use numeric scores with 2 decimals.
"""
    return prompt

def build_breaking_prompt(breaking_macro, breaking_gold, tech):
    bm = "\n".join([f"- {h['title']}" for h in breaking_macro])
    bg = "\n".join([f"- {h['title']}" for h in breaking_gold])
    prompt = f"""
You are a crisis-sensitive market analyst. Assess the immediate impact of the BREAKING headlines below on gold price.
Return strictly valid JSON only with:
- breaking_impact: one of ["high","medium","low","none"]
- direction: one of ["bullish","bearish","neutral"] (effect on gold)
- impact_score: float between -1.0 and 1.0 (negative => bearish)
- urgency: one of ["immediate","hours","days"]
- reasoning: one-sentence justification (max 20 words)

Breaking Macro headlines:
{bm}

Breaking Gold headlines:
{bg}

Current Gold technical: price ${tech['price'] if tech else 'N/A'}, support ${tech['support'] if tech else 'N/A'}, resistance ${tech['resistance'] if tech else 'N/A'}
"""
    return prompt

# ---------------------
# Main pipeline
# ---------------------
def main():
    logger.info("=== GOLD SIGNAL SYSTEM v5 (fixed) ===")

    # Step 1: Fetch data
    fetcher = DataFetcher()
    gold_4h = fetcher.get_gold_4h()

    # Key indices (from FRED) - these IDs are common; change as needed
    indices = {
        'sp500': fetcher.get_fred_latest('SP500'),
        'vix': fetcher.get_fred_latest('VIXCLS'),
        'fed_rate': fetcher.get_fred_latest('DFF'),
        'yield_10y': fetcher.get_fred_latest('DGS10'),
    }
    fear_greed = fetcher.get_fear_and_greed()
    logger.info("Gold latest: %s", gold_4h['close'].iloc[-1] if (gold_4h is not None and 'close' in gold_4h.columns) else 'N/A')
    logger.info("Indices: %s", indices)
    logger.info("Fear&Greed: %s", fear_greed['value'] if fear_greed else 'N/A')

    # Step 2: News
    nf = NewsFetcher()
    news = nf.fetch_all()

    # Step 3: Filter headlines semantically
    macro_regular, gold_regular = filter_headlines(news.get('regular', []))
    macro_breaking, gold_breaking = filter_headlines(news.get('breaking', []))
    logger.info("Filtered - Regular macro:%d gold:%d | Breaking macro:%d gold:%d",
                len(macro_regular), len(gold_regular), len(macro_breaking), len(gold_breaking))

    # Step 4: Technicals
    tech = calculate_technicals(gold_4h)

    # Step 5: LLM analysis (regular)
    regular_analysis = {}
    if Config.GROQ_API_KEY:
        prompt_reg = build_regular_prompt(indices, fear_greed or {}, macro_regular, gold_regular, tech)
        logger.info("Calling GROQ for regular analysis...")
        reg_out = groq_call(prompt_reg, temperature=0.15)
        regular_analysis = parse_json_block(reg_out) if reg_out else {}
        logger.info("Regular LLM output: %s", regular_analysis)
    else:
        logger.info("GROQ not configured; skipping LLM regular analysis")

    # Step 6: LLM analysis (breaking)
    if (macro_breaking or gold_breaking) and Config.GROQ_API_KEY:
        prompt_break = build_breaking_prompt(macro_breaking, gold_breaking, tech)
        logger.info("Calling GROQ for breaking analysis...")
        brk_out = groq_call(prompt_break, temperature=0.0)
        breaking_analysis = parse_json_block(brk_out) if brk_out else {}
        logger.info("Breaking LLM output: %s", breaking_analysis)
    else:
        breaking_analysis = {'breaking_impact': 'none', 'impact_score': 0}

    # Step 7: Merge signals (weighted)
    bw_map = {'high': 0.6, 'medium': 0.3, 'low': 0.1, 'none': 0}
    bw = bw_map.get(breaking_analysis.get('breaking_impact', 'none'), 0)
    reg_score = float(regular_analysis.get('gold_score', 0)) if regular_analysis else 0.0
    brk_score = float(breaking_analysis.get('impact_score', 0)) if breaking_analysis else 0.0
    final_gold_score = (1 - bw) * reg_score + bw * brk_score
    final_sentiment = "BULLISH" if final_gold_score > 0.25 else "BEARISH" if final_gold_score < -0.25 else "NEUTRAL"

    logger.info("Scores -> regular: %.3f | breaking: %.3f (w=%.2f) | final: %.3f => %s",
                reg_score, brk_score, bw, final_gold_score, final_sentiment)

    # Step 8: Trading levels (simple logic)
    if tech and final_sentiment != "NEUTRAL":
        if final_sentiment == "BULLISH":
            entry = max(tech['support'] + tech['atr'], tech['price'] * 0.998)
            tp1 = tech['resistance_2']
            tp2 = tech['resistance']
            sl = tech['support'] - tech['atr'] * 0.5
        else:
            entry = min(tech['resistance'] - tech['atr'], tech['price'] * 1.002)
            tp1 = tech['support_2']
            tp2 = tech['support']
            sl = tech['resistance'] + tech['atr'] * 0.5
        rr = abs(tp1 - entry) / (abs(entry - sl) if abs(entry - sl) > 1e-9 else 1e-9)
        position_size = '50%' if abs(final_gold_score) > 0.5 else '25%'
        signal = {'signal': 'BUY' if final_sentiment == 'BULLISH' else 'SELL',
                  'entry': round(entry, 2), 'tp1': round(tp1, 2), 'tp2': round(tp2, 2), 'sl': round(sl, 2),
                  'risk_reward': round(rr, 2), 'position_size': position_size}
    else:
        signal = {'signal': 'HOLD', 'entry': None, 'tp1': None, 'tp2': None, 'sl': None, 'risk_reward': None, 'position_size': '0%'}

    # Step 9: Output
    output = {
        'timestamp': now_utc().isoformat(),
        'indices': indices,
        'fear_greed': fear_greed,
        'regular_analysis': regular_analysis,
        'breaking_analysis': breaking_analysis,
        'scores': {'regular': reg_score, 'breaking': brk_score, 'final': round(final_gold_score, 3)},
        'technical': tech,
        'signal': signal,
        'headlines': {'breaking': macro_breaking + gold_breaking, 'regular': macro_regular + gold_regular}
    }
    safe_json_dump(Config.OUTPUT_FILE, output)
    logger.info("✓ Saved output to %s", Config.OUTPUT_FILE)
    logger.info("Signal: %s", signal)

if __name__ == "__main__":
    main()