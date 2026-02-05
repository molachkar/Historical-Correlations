import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time
import re
from collections import defaultdict

# -------- CONFIG --------
OUTPUT_FILE = "C:\\Users\\PC\\Desktop\\MML\\gold_news_analysis.json"

NEWSAPI_KEY = "07d638496f4741849ce55d2c42320107"
GNEWS_KEY = "3ca29aea34716d7b94e513c9463ef5f1"

# Source credibility weights
SOURCE_WEIGHTS = {
    'Reuters': 1.0,
    'Bloomberg': 1.0,
    'Financial Times': 0.95,
    'WSJ': 0.95,
    'CNBC': 0.85,
    'Investing.com': 0.80,
    'Kitco': 0.80,
    'MarketWatch': 0.75,
    'Yahoo Finance': 0.70,
    'GNews': 0.60,
    'NewsAPI': 0.60,
    'Google News': 0.55,
    'Bing News': 0.50
}

# -------- DYNAMIC GOLD CONTEXT --------
def build_gold_context():
    """Dynamic context that captures price drivers"""
    return """
    Gold (XAU/USD) precious metal commodity asset trading.
    
    Primary price drivers:
    - US Dollar (DXY) inverse correlation: stronger USD = lower gold
    - Real interest rates: higher rates = lower gold appeal
    - Federal Reserve monetary policy: rate cuts support gold, hikes pressure it
    - Inflation expectations: rising inflation = gold hedge demand
    - Geopolitical tensions: uncertainty drives safe-haven demand
    - Central bank purchases and reserves management
    - Technical levels: support resistance breakouts consolidation
    
    Market signals:
    Bullish: rate cuts, USD weakness, inflation fears, geopolitical crisis, safe-haven flows
    Bearish: rate hikes, USD strength, risk-on sentiment, opportunity cost from bonds
    
    Related terms: bullion, spot gold, gold futures, GLD ETF, mining stocks (Barrick, Newmont),
    jewelry demand, physical demand, paper gold, gold standard, monetary metal
    """

# -------- LOAD MODEL --------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
gold_embedding = model.encode(build_gold_context())
print("✓ Model loaded\n")

# -------- HEADLINE CLEANING --------
def clean_headline(text):
    """Remove noise, normalize text"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special chars but keep basic punctuation
    text = re.sub(r'[^\w\s\-\.\,\:\;\$\%]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common filler
    removals = [
        'Read more', 'Click here', 'Subscribe', 'Advertisement',
        'Sponsored', 'Breaking:', 'BREAKING:', 'UPDATE:', 'Update:'
    ]
    for phrase in removals:
        text = text.replace(phrase, '')
    
    return text.strip()

# -------- DEDUPLICATION --------
def deduplicate_exact(articles):
    """Remove exact duplicate URLs and titles"""
    seen_urls = set()
    seen_titles = set()
    unique = []
    
    for article in articles:
        url = article.get('link', '').strip()
        title = article.get('title', '').strip().lower()
        
        if not url or not title:
            continue
        
        if url not in seen_urls and title not in seen_titles:
            seen_urls.add(url)
            seen_titles.add(title)
            unique.append(article)
    
    return unique

# -------- NEWS SOURCES (IMPROVED) --------

def get_newsapi_gold():
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': '(gold OR XAUUSD OR XAU/USD) AND (price OR trading OR market)',
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'from': (datetime.now() - timedelta(hours=48)).strftime('%Y-%m-%d'),
            'pageSize': 100
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        articles = []
        for item in data.get('articles', []):
            try:
                published = datetime.strptime(item.get('publishedAt', '')[:19], '%Y-%m-%dT%H:%M:%S')
            except:
                published = datetime.now()
            
            source_name = item.get('source', {}).get('name', 'Unknown')
            
            articles.append({
                'source': source_name,
                'title': clean_headline(item.get('title', '')),
                'link': item.get('url', ''),
                'published': published,
                'summary': clean_headline(item.get('description', '') or ''),
                'raw_source': 'NewsAPI'
            })
        return articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

def get_investing_com_gold():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.investing.com/commodities/gold-news"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Try multiple selectors
        selectors = [
            ('article', {}),
            ('div', {'class': 'largeTitle'}),
            ('div', {'class': 'textDiv'})
        ]
        
        for tag, attrs in selectors:
            items = soup.find_all(tag, attrs, limit=30)
            for item in items:
                try:
                    title_tag = item.find('a')
                    if not title_tag:
                        continue
                    
                    title = clean_headline(title_tag.get_text())
                    link = title_tag.get('href', '')
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.investing.com' + link
                    
                    if title and len(title) > 10:
                        articles.append({
                            'source': 'Investing.com',
                            'title': title,
                            'link': link,
                            'published': datetime.now(),
                            'summary': '',
                            'raw_source': 'Investing.com'
                        })
                except:
                    continue
        
        return articles
    except Exception as e:
        print(f"Investing.com error: {e}")
        return []

def get_reuters_gold():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.reuters.com/markets/commodities/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        links = soup.find_all('a', href=True, limit=100)
        
        for link in links:
            try:
                title = clean_headline(link.get_text())
                url = link['href']
                
                if not title or len(title) < 15:
                    continue
                
                # Filter for gold-related
                if not any(kw in title.lower() for kw in ['gold', 'xau', 'precious', 'bullion']):
                    continue
                
                if not url.startswith('http'):
                    url = 'https://www.reuters.com' + url
                
                articles.append({
                    'source': 'Reuters',
                    'title': title,
                    'link': url,
                    'published': datetime.now(),
                    'summary': '',
                    'raw_source': 'Reuters'
                })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"Reuters error: {e}")
        return []

def get_kitco_news():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.kitco.com/news/gold/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Multiple selectors
        selectors = [
            ('div', {'class': 'top-news-item'}),
            ('div', {'class': 'news-item'}),
            ('article', {})
        ]
        
        for tag, attrs in selectors:
            items = soup.find_all(tag, attrs, limit=30)
            for item in items:
                try:
                    title_tag = item.find('a')
                    if not title_tag:
                        continue
                    
                    title = clean_headline(title_tag.get_text())
                    link = title_tag.get('href', '')
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.kitco.com' + link
                    
                    if title and len(title) > 10:
                        articles.append({
                            'source': 'Kitco',
                            'title': title,
                            'link': link,
                            'published': datetime.now(),
                            'summary': '',
                            'raw_source': 'Kitco'
                        })
                except:
                    continue
        
        return articles
    except Exception as e:
        print(f"Kitco error: {e}")
        return []

def get_cnbc_gold():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.cnbc.com/commodities/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        links = soup.find_all('a', href=True, limit=100)
        
        for link in links:
            try:
                title = clean_headline(link.get_text())
                url = link.get('href', '')
                
                if not title or len(title) < 15:
                    continue
                
                if not any(kw in title.lower() for kw in ['gold', 'xau', 'precious', 'metals']):
                    continue
                
                if url and 'cnbc.com' not in url and not url.startswith('http'):
                    url = 'https://www.cnbc.com' + url
                
                articles.append({
                    'source': 'CNBC',
                    'title': title,
                    'link': url,
                    'published': datetime.now(),
                    'summary': '',
                    'raw_source': 'CNBC'
                })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"CNBC error: {e}")
        return []

def get_marketwatch_gold():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.marketwatch.com/investing/future/gold"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        links = soup.find_all('a', href=True, limit=100)
        
        for link in links:
            try:
                title = clean_headline(link.get_text())
                url = link.get('href', '')
                
                if not title or len(title) < 15:
                    continue
                
                if not url.startswith('http') and url:
                    url = 'https://www.marketwatch.com' + url
                
                articles.append({
                    'source': 'MarketWatch',
                    'title': title,
                    'link': url,
                    'published': datetime.now(),
                    'summary': '',
                    'raw_source': 'MarketWatch'
                })
            except:
                continue
        
        return articles[:25]
    except Exception as e:
        print(f"MarketWatch error: {e}")
        return []

# -------- COLLECT --------
def collect_all_news():
    print("="*60)
    print("COLLECTING GOLD NEWS")
    print("="*60)
    
    all_articles = []
    
    sources = [
        ("Reuters", get_reuters_gold),
        ("Investing.com", get_investing_com_gold),
        ("Kitco", get_kitco_news),
        ("CNBC", get_cnbc_gold),
        ("MarketWatch", get_marketwatch_gold),
        ("NewsAPI", get_newsapi_gold)
    ]
    
    for i, (name, func) in enumerate(sources, 1):
        print(f"{i}. {name}...")
        try:
            articles = func()
            print(f"   {len(articles)} articles")
            all_articles.extend(articles)
            time.sleep(2)
        except Exception as e:
            print(f"   Error: {e}")
    
    # Exact deduplication
    all_articles = deduplicate_exact(all_articles)
    
    print(f"\n✓ Total: {len(all_articles)}\n")
    return all_articles

# -------- RELEVANCE FILTER --------
def filter_relevant(articles, threshold=0.30):
    print("="*60)
    print("FILTERING RELEVANCE")
    print("="*60)
    
    relevant = []
    
    for article in articles:
        text = f"{article['title']} {article.get('summary', '')}"
        
        if not text.strip() or len(text) < 15:
            continue
        
        try:
            emb = model.encode(text)
            similarity = cosine_similarity([gold_embedding], [emb])[0][0]
            
            if similarity >= threshold:
                article['relevance'] = float(similarity)
                relevant.append(article)
        except:
            continue
    
    relevant.sort(key=lambda x: x['relevance'], reverse=True)
    
    print(f"✓ {len(relevant)} relevant\n")
    return relevant

# -------- CLUSTERING --------
def cluster_headlines(articles, threshold=0.80):
    print("="*60)
    print("CLUSTERING DUPLICATES")
    print("="*60)
    print(f"Before: {len(articles)}")
    
    if len(articles) < 2:
        return articles
    
    try:
        texts = [f"{a['title']} {a.get('summary', '')}" for a in articles]
        embeddings = model.encode(texts)
        
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = np.clip(1 - sim_matrix, 0, None)
        
        clustering = DBSCAN(eps=1-threshold, min_samples=1, metric='precomputed')
        clusters = clustering.fit_predict(dist_matrix)
        
        unique = []
        for cluster_id in set(clusters):
            cluster_articles = [articles[i] for i in range(len(articles)) if clusters[i] == cluster_id]
            
            # Pick best by relevance * source weight
            best = max(cluster_articles, key=lambda x: x.get('relevance', 0) * SOURCE_WEIGHTS.get(x['source'], 0.5))
            
            best['duplicate_count'] = len(cluster_articles)
            best['cluster_sources'] = list(set(a['source'] for a in cluster_articles))
            
            unique.append(best)
        
        print(f"After: {len(unique)}")
        print(f"Removed: {len(articles) - len(unique)}\n")
        
        return unique
    
    except Exception as e:
        print(f"Clustering error: {e}\n")
        return articles

# -------- SENTIMENT (KEYWORD-BASED) --------
def classify_sentiment(articles):
    print("="*60)
    print("SENTIMENT CLASSIFICATION")
    print("="*60)
    
    bullish = [
        'rally', 'surge', 'gain', 'rise', 'climb', 'jump', 'soar', 'breakout', 'bullish',
        'support', 'strength', 'buying', 'safe haven', 'safehaven', 'record', 'inflation',
        'rate cut', 'cuts rates', 'dovish', 'weaker dollar', 'tensions', 'boost', 'higher',
        'uptrend', 'demand', 'haven', 'hedge', 'buy'
    ]
    
    bearish = [
        'fall', 'drop', 'decline', 'plunge', 'tumble', 'sell-off', 'selloff', 'bearish',
        'resistance', 'weakness', 'selling', 'profit taking', 'rate hike', 'hawkish',
        'stronger dollar', 'lower', 'correction', 'pressure', 'slump', 'downtrend', 'sell'
    ]
    
    for article in articles:
        text = f"{article['title']} {article.get('summary', '')}".lower()
        
        bull_score = sum(1 for kw in bullish if kw in text)
        bear_score = sum(1 for kw in bearish if kw in text)
        
        if bull_score > bear_score and bull_score >= 1:
            article['sentiment'] = 'BULLISH'
            article['sentiment_strength'] = bull_score - bear_score
        elif bear_score > bull_score and bear_score >= 1:
            article['sentiment'] = 'BEARISH'
            article['sentiment_strength'] = bear_score - bull_score
        else:
            article['sentiment'] = 'NEUTRAL'
            article['sentiment_strength'] = 0
    
    bullish_count = sum(1 for a in articles if a['sentiment'] == 'BULLISH')
    bearish_count = sum(1 for a in articles if a['sentiment'] == 'BEARISH')
    neutral_count = sum(1 for a in articles if a['sentiment'] == 'NEUTRAL')
    
    print(f"BULLISH: {bullish_count} | BEARISH: {bearish_count} | NEUTRAL: {neutral_count}\n")
    
    return articles

# -------- WEIGHTED SENTIMENT SCORE --------
def calculate_aggregate_sentiment(articles):
    print("="*60)
    print("AGGREGATE SENTIMENT")
    print("="*60)
    
    total_weight = 0
    weighted_sentiment = 0
    
    for article in articles:
        if article['sentiment'] == 'NEUTRAL':
            continue
        
        # Recency decay (last 24h = 1.0, older = decay)
        hours_old = (datetime.now() - article['published']).total_seconds() / 3600
        recency_factor = max(0.3, 1.0 - (hours_old / 48))
        
        # Source credibility
        source_weight = SOURCE_WEIGHTS.get(article['source'], 0.5)
        
        # Sentiment direction
        direction = 1 if article['sentiment'] == 'BULLISH' else -1
        
        # Combined weight
        weight = article['relevance'] * source_weight * recency_factor * article.get('duplicate_count', 1)
        
        article['calculated_weight'] = round(weight, 3)
        
        total_weight += weight
        weighted_sentiment += direction * weight * article['sentiment_strength']
    
    if total_weight == 0:
        final_sentiment = 0
    else:
        final_sentiment = weighted_sentiment / total_weight
    
    print(f"Aggregate Sentiment Score: {final_sentiment:.3f}")
    
    if final_sentiment > 0.3:
        interpretation = "STRONGLY BULLISH"
    elif final_sentiment > 0.1:
        interpretation = "MODERATELY BULLISH"
    elif final_sentiment < -0.3:
        interpretation = "STRONGLY BEARISH"
    elif final_sentiment < -0.1:
        interpretation = "MODERATELY BEARISH"
    else:
        interpretation = "NEUTRAL / MIXED"
    
    print(f"Interpretation: {interpretation}\n")
    
    return final_sentiment, interpretation

# -------- MAIN --------
def main():
    print("\n🥇 GOLD NEWS ANALYZER v2.0")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    articles = collect_all_news()
    
    if not articles:
        print("No articles collected")
        return
    
    relevant = filter_relevant(articles, threshold=0.30)
    
    if not relevant:
        print("No relevant articles")
        return
    
    deduplicated = cluster_headlines(relevant, threshold=0.80)
    
    classified = classify_sentiment(deduplicated)
    
    aggregate_score, interpretation = calculate_aggregate_sentiment(classified)
    
    # Filter out neutrals for display
    final = [a for a in classified if a['sentiment'] != 'NEUTRAL']
    
    print("="*60)
    print("TOP HEADLINES")
    print("="*60)
    
    for i, article in enumerate(final[:15], 1):
        icon = "📈" if article['sentiment'] == 'BULLISH' else "📉"
        dup = f"[{article['duplicate_count']}x]" if article.get('duplicate_count', 1) > 1 else ""
        weight = article.get('calculated_weight', 0)
        
        print(f"\n{i}. {icon} {article['sentiment']} {dup}")
        print(f"   {article['title']}")
        print(f"   {article['source']} | Rel: {article['relevance']:.2f} | Wt: {weight:.3f}")
        if article.get('cluster_sources'):
            print(f"   Also: {', '.join(article['cluster_sources'][:4])}")
    
    # Save
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'aggregate_sentiment_score': round(aggregate_score, 3),
        'interpretation': interpretation,
        'stats': {
            'total_collected': len(articles),
            'relevant': len(relevant),
            'after_dedup': len(deduplicated),
            'bullish': sum(1 for a in classified if a['sentiment'] == 'BULLISH'),
            'bearish': sum(1 for a in classified if a['sentiment'] == 'BEARISH'),
            'neutral': sum(1 for a in classified if a['sentiment'] == 'NEUTRAL')
        },
        'articles': final
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Saved: {OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()