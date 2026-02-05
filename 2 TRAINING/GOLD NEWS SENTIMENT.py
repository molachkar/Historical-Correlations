import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ta
from dateutil import parser as date_parser
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ===================================================
# CONFIGURATION
# ===================================================
class Config:
    TWELVE_DATA_KEY = "6195d22ee1754b65ac74dd4464f8d936"
    FRED_API_KEY = "a7a7e3fcfe30edeed9edc26dbea210b7"
    GROQ_API_KEY = "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"
    NEWSAPI_KEY = "07d638496f4741849ce55d2c42320107"

    BASE_DIR = Path("C://Users//PC//Desktop")
    CACHE_DIR = BASE_DIR / "cache"
    OUTPUT_FILE = BASE_DIR / "gold_signal_v5.json"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_CACHE = CACHE_DIR / "fixed_vectors.npz"
    HEADLINE_CACHE = CACHE_DIR / "seen_headlines.json"
    DATA_CACHE = CACHE_DIR / "data_cache.json"


# ===================================================
# FIXED DESCRIPTIONS
# ===================================================
MACRO_DESC = """
Federal Reserve monetary policy decisions and interest rate changes affecting dollar strength and Treasury yields.
Economic data releases including inflation reports, employment statistics, and GDP growth impacting market sentiment.
Stock market performance across major indices reflecting risk appetite and investor positioning.
Geopolitical developments and international tensions influencing safe-haven flows and currency movements.
Central bank actions and fiscal policy decisions shaping liquidity conditions and asset valuations.
"""

GOLD_DESC = """
Gold price movements driven by dollar strength, real interest rates, and inflation expectations.
Safe-haven demand responding to geopolitical tensions, market volatility, and economic uncertainty.
Central bank gold purchases and reserve diversification affecting physical demand dynamics.
Investment flows through ETFs and futures contracts reflecting institutional positioning changes.
Mining supply constraints and jewelry consumption patterns in major markets like India and China.
"""

# ===================================================
# EMBEDDING MODEL & VECTOR CREATION
# ===================================================
class VectorManager:
    @staticmethod
    def load_or_create_vectors(model):
        if Config.VECTOR_CACHE.exists():
            logging.info("Loading cached vectors...")
            data = np.load(Config.VECTOR_CACHE)
            return data['macro'], data['gold']
        logging.info("Creating fixed vectors (first-time only)...")
        macro_vec = model.encode(MACRO_DESC)
        gold_vec = model.encode(GOLD_DESC)
        np.savez(Config.VECTOR_CACHE, macro=macro_vec, gold=gold_vec)
        return macro_vec, gold_vec


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
MACRO_VEC, GOLD_VEC = VectorManager.load_or_create_vectors(embedding_model)


# ===================================================
# DATA FETCHER
# ===================================================
class DataFetcher:
    def __init__(self):
        self.cache_file = Config.DATA_CACHE
        if not self.cache_file.exists():
            json.dump({}, open(self.cache_file, 'w'))

    def _load_cache(self):
        try:
            return json.load(open(self.cache_file))
        except:
            return {}

    def _save_cache(self, data):
        json.dump(data, open(self.cache_file, 'w'), default=str)

    def _cache(self, key, value):
        cache = self._load_cache()
        cache[key] = {"data": value, "time": datetime.utcnow().isoformat()}
        self._save_cache(cache)

    def _load(self, key, max_hours=24):
        cache = self._load_cache()
        if key in cache:
            try:
                age = (datetime.utcnow() - date_parser.isoparse(cache[key]['time'])).total_seconds() / 3600
                if age < max_hours:
                    return cache[key]['data']
            except:
                return cache[key]['data']
        return None

    def get_gold_4h(self):
        cached = self._load("gold_4h", 1)
        if cached:
            return pd.DataFrame(cached)
        try:
            r = requests.get("https://api.twelvedata.com/time_series", params={
                "symbol": "XAU/USD", "interval": "4h",
                "apikey": Config.TWELVE_DATA_KEY, "outputsize": 42
            }, timeout=15)
            j = r.json()
            if "values" in j:
                df = pd.DataFrame(j["values"])
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna().sort_values('datetime')
                self._cache("gold_4h", df.to_dict('records'))
                return df
        except:
            logging.exception("Gold data fetch error")
        return self._load("gold_4h", 72)

    def get_fred(self, series_id):
        cached = self._load(f'fred_{series_id}', 12)
        if cached:
            return cached
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", params={
                "series_id": series_id, "api_key": Config.FRED_API_KEY,
                "file_type": "json", "sort_order": "desc", "limit": 1
            }, timeout=10)
            obs = r.json().get('observations', [])
            if obs:
                val = obs[0].get('value')
                value = float(val) if val not in (None, ".", "") else None
                self._cache(f'fred_{series_id}', value)
                return value
        except:
            logging.exception("FRED fetch error")
        return self._load(f'fred_{series_id}', 72)

    def get_fear_greed(self):
        cached = self._load("fear_greed", 6)
        if cached:
            return cached
        try:
            r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", timeout=10)
            raw = r.json()
            data = raw.get('fear_and_greed') or raw.get('data', {}).get('fear_and_greed', {})
            if not data:
                logging.warning("Fear & Greed structure changed: keys %s", list(raw.keys()))
            result = {
                'value': data.get('score'),
                'rating': data.get('rating'),
                'normalized': (data.get('score', 50) - 50) / 50 if data.get('score') else 0
            }
            self._cache("fear_greed", result)
            return result
        except:
            logging.exception("Fear & Greed fetch error")
        return self._load("fear_greed", 24)


# ===================================================
# NEWS FETCHER
# ===================================================
class NewsFetcher:
    def __init__(self):
        try:
            self.seen = json.load(open(Config.HEADLINE_CACHE))
        except:
            self.seen = {}

    def _mark_seen(self, title):
        self.seen[hashlib.md5(title.lower().encode()).hexdigest()] = datetime.utcnow().isoformat()
        json.dump(self.seen, open(Config.HEADLINE_CACHE, 'w'))

    def _is_seen(self, title):
        h = hashlib.md5(title.lower().encode()).hexdigest()
        if h in self.seen:
            try:
                age = (datetime.utcnow() - date_parser.isoparse(self.seen[h])).total_seconds() / 3600
                return age < 12
            except:
                return True
        return False

    def fetch_newsapi(self, query, hours):
        try:
            r = requests.get("https://newsapi.org/v2/everything", params={
                "q": query, "language": "en", "sortBy": "publishedAt",
                "apiKey": Config.NEWSAPI_KEY, "from": (datetime.utcnow() - timedelta(hours=hours)).isoformat(),
                "pageSize": 100
            }, timeout=15)
            headlines = []
            for a in r.json().get("articles", []):
                title = a.get("title", "").strip()
                if len(title) < 15 or self._is_seen(title):
                    continue
                try:
                    pub = date_parser.isoparse(a["publishedAt"])
                    age = (datetime.utcnow() - pub).total_seconds() / 3600
                except:
                    age = 999
                category = "breaking" if age < 3 else "recent" if age < 24 else "background"
                headlines.append({"title": title, "source": a["source"]["name"], "age_hours": round(age, 1), "category": category})
                self._mark_seen(title)
            time.sleep(1)
            return headlines
        except:
            logging.exception("NewsAPI error")
            return []

    def fetch_all(self):
        logging.info("Fetching news from all sources...")
        all_h = []
        # === Include CPI & inflation in query ===
        all_h.extend(self.fetch_newsapi(
            '("Federal Reserve" OR "interest rates" OR "stock market" OR inflation OR CPI OR "consumer price index" OR gold OR XAUUSD)', 168
        ))
        rss_sources = [
            ('https://www.kitco.com/rss/KitcoNews-GoldSilver.xml', 'Kitco', 48),
            ('https://feeds.bloomberg.com/markets/news.rss', 'Bloomberg', 72),
            ('https://news.google.com/rss/search?q=Federal+Reserve+gold&hl=en', 'Google News', 24),
        ]
        import feedparser
        for url, name, hrs in rss_sources:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:50]:
                    title = e.get("title", "").strip()
                    if len(title) < 15 or self._is_seen(title):
                        continue
                    try:
                        pub = date_parser.parse(e.published)
                        age = (datetime.utcnow() - pub).total_seconds() / 3600
                    except:
                        age = 999
                    if age > hrs:
                        continue
                    cat = "breaking" if age < 3 else "recent"
                    all_h.append({"title": title, "source": name, "age_hours": round(age, 1), "category": cat})
                    self._mark_seen(title)
            except:
                continue
        breaking = [h for h in all_h if h["category"] == "breaking"]
        regular = [h for h in all_h if h["category"] != "breaking"]
        logging.info(f"Collected {len(all_h)} headlines (Breaking: {len(breaking)}, Regular: {len(regular)})")
        return {"breaking": breaking, "regular": regular}


# ===================================================
# FILTER HEADLINES
# ===================================================
def filter_headlines(headlines, threshold=0.28):
    macro, gold = [], []
    for h in headlines:
        vec = embedding_model.encode(h['title'])
        m_sim = float(cosine_similarity([MACRO_VEC], [vec])[0][0])
        g_sim = float(cosine_similarity([GOLD_VEC], [vec])[0][0])
        if m_sim > threshold:
            h['relevance'] = m_sim
            macro.append(h)
        if g_sim > threshold + 0.02:
            h['relevance'] = g_sim
            gold.append(h)
    return macro, gold


# ===================================================
# TECHNICAL ANALYSIS
# ===================================================
def calculate_technicals(df):
    if df is None or len(df) < 20:
        return None
    df = df.copy()
    for c in ['open', 'high', 'low', 'close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    rsi = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    cci = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    bb = ta.volatility.BollingerBands(df['close'])
    ema20 = df['close'].ewm(span=20).mean()
    ema50 = df['close'].ewm(span=50).mean() if len(df) >= 50 else ema20
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    recent = df.tail(28)
    return {
        'price': float(df['close'].iloc[-1]),
        'rsi': float(rsi.iloc[-1]),
        'stoch_k': float(stoch.stoch().iloc[-1]),
        'cci': float(cci.iloc[-1]),
        'bb_pct': float((df['close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) /
                        (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])),
        'ema20': float(ema20.iloc[-1]),
        'ema50': float(ema50.iloc[-1]),
        'atr': float(atr.iloc[-1]),
        'resistance': float(recent['high'].max()),
        'support': float(recent['low'].min()),
        'resistance_2': float(recent['high'].nlargest(3).iloc[-1]),
        'support_2': float(recent['low'].nsmallest(3).iloc[-1]),
        'trend': 'Uptrend' if df['close'].iloc[-1] > ema20.iloc[-1] else 'Downtrend'
    }


# ===================================================
# GROQ LLM WRAPPER
# ===================================================
def groq(prompt, temp=0.3):
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": temp,
                  "max_tokens": 1024},
            timeout=30)
        return r.json()['choices'][0]['message']['content']
    except:
        logging.exception("Groq API call failed")
        return None


def parse_json(text):
    if not text:
        return {}
    try:
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        return json.loads(text.strip())
    except:
        return {}


# ===================================================
# MAIN
# ===================================================
def main():
    print("\n============================================================")
    print("🥇 GOLD + MARKET SENTIMENT SYSTEM v5.5")
    print("============================================================\n")

    fetcher = DataFetcher()
    gold_4h = fetcher.get_gold_4h()
    indices = {'sp500': fetcher.get_fred('SP500'),
               'vix': fetcher.get_fred('VIXCLS'),
               'fed_rate': fetcher.get_fred('DFF'),
               'yield_10y': fetcher.get_fred('DGS10')}
    fear_greed = fetcher.get_fear_greed()

    print(f"✓ Gold: ${gold_4h['close'].iloc[-1] if gold_4h is not None else 'N/A'}")
    print(f"✓ Fear & Greed: {fear_greed if fear_greed else 'Unavailable'}\n")

    news_fetcher = NewsFetcher()
    news = news_fetcher.fetch_all()
    macro_regular, gold_regular = filter_headlines(news['regular'])
    macro_breaking, gold_breaking = filter_headlines(news['breaking'])

    print("\n===== MACRO HEADLINES (Top 10) =====")
    for h in macro_regular[:10]:
        print(f"[{h['relevance']:.2f}] {h['title']}")

    print("\n===== GOLD HEADLINES (Top 10) =====")
    for h in gold_regular[:10]:
        print(f"[{h['relevance']:.2f}] {h['title']}")

    tech = calculate_technicals(gold_4h)
    print(f"\nTechnical snapshot: {tech}\n")

    prompt = f"""Market Data: S&P {indices['sp500']}, VIX {indices['vix']}, FearGreed {fear_greed.get('value') if fear_greed else 'N/A'}

Macro Headlines:
{chr(10).join(['- '+h['title'] for h in macro_regular[:15]])}

Gold Headlines:
{chr(10).join(['- '+h['title'] for h in gold_regular[:10]])}

Gold Technicals: {tech}
Output JSON:
{{"market_sentiment": "risk_on/risk_off", "market_score": -1 to 1, "gold_sentiment": "BULLISH/BEARISH/NEUTRAL", "gold_score": -1 to 1, "summary": "2 sentences"}}"""
    raw = groq(prompt)
    print("\n===== GROQ RAW RESPONSE =====\n", raw)
    result = parse_json(raw)
    print("\n===== GROQ SUMMARY =====\n", result.get('summary', 'No summary returned'))

    json.dump(result, open(Config.OUTPUT_FILE, 'w'), indent=2)
    print(f"\n✓ Saved: {Config.OUTPUT_FILE}")


if __name__ == "__main__":
    main()