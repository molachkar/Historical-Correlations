import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time

# -------- CONFIG --------
OUTPUT_FILE = "C:\\Users\\PC\\Desktop\\MML\\gold_news_analysis.json"

# API KEYS
NEWSAPI_KEY = "07d638496f4741849ce55d2c42320107"
GNEWS_KEY = "3ca29aea34716d7b94e513c9463ef5f1"

# -------- GOLD KEYWORDS EMBEDDING --------
gold_keywords_text = """
Gold, XAUUSD, XAU/USD, precious metals, gold price, gold market, bullion, 
gold trading, spot gold, gold futures, gold ETF, GLD, gold reserves, 
central bank gold, gold demand, gold supply, jewelry demand, 
safe haven asset, inflation hedge, monetary policy impact on gold,
Federal Reserve gold, interest rates gold correlation, USD gold inverse,
geopolitical tensions gold, recession gold, economic uncertainty gold,
gold mining stocks, Barrick Gold, Newmont, gold production,
gold technical analysis, gold resistance support levels,
gold breakout, gold rally, gold decline, gold volatility
"""

# -------- LOAD EMBEDDING MODEL --------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
gold_embedding = model.encode(gold_keywords_text)
print("✓ Gold keywords embedded\n")

# -------- NEWS SOURCES --------

def get_yfinance_news():
    """Yahoo Finance - multiple tickers"""
    try:
        import yfinance as yf
        articles = []
        tickers = ["GC=F", "GLD", "GOLD", "NEM", "RGLD"]
        
        for ticker_symbol in tickers:
            try:
                ticker = yf.Ticker(ticker_symbol)
                news = ticker.news
                for item in news[:5]:
                    articles.append({
                        'source': f'Yahoo Finance ({ticker_symbol})',
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                        'summary': item.get('summary', '')
                    })
                time.sleep(1)
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return []

def get_newsapi_gold():
    """NewsAPI"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'gold OR XAUUSD OR "precious metals" OR bullion',
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
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
                
            articles.append({
                'source': f"NewsAPI - {item.get('source', {}).get('name', 'Unknown')}",
                'title': item.get('title', ''),
                'link': item.get('url', ''),
                'published': published,
                'summary': item.get('description', '') or ''
            })
        return articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

def get_gnews_api():
    """GNews API - Fixed date parsing"""
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            'q': 'gold OR XAUUSD OR "precious metals"',
            'lang': 'en',
            'country': 'us',
            'max': 100,
            'apikey': GNEWS_KEY,
            'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        articles = []
        for item in data.get('articles', []):
            published_str = item.get('publishedAt', '')
            try:
                # Try with milliseconds
                published = datetime.strptime(published_str[:26], '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                # Try without milliseconds
                try:
                    published = datetime.strptime(published_str[:19], '%Y-%m-%dT%H:%M:%S')
                except:
                    published = datetime.now()
            
            articles.append({
                'source': f"GNews - {item.get('source', {}).get('name', 'Unknown')}",
                'title': item.get('title', ''),
                'link': item.get('url', ''),
                'published': published,
                'summary': item.get('description', '') or ''
            })
        return articles
    except Exception as e:
        print(f"GNews error: {e}")
        return []

def get_google_news():
    """Google News RSS"""
    try:
        import feedparser
        url = "https://news.google.com/rss/search?q=gold+XAUUSD+price&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        
        articles = []
        for entry in feed.entries[:50]:
            try:
                published = datetime(*entry.get('published_parsed', datetime.now().timetuple())[:6])
            except:
                published = datetime.now()
                
            articles.append({
                'source': 'Google News',
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': published,
                'summary': entry.get('summary', '') or ''
            })
        
        return articles
    except Exception as e:
        print(f"Google News error: {e}")
        return []

def get_bing_news():
    """Bing News Search"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f"https://www.bing.com/news/search?q=gold+price+XAUUSD&qft=interval%3d%227%22&form=PTFTNR"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        news_cards = soup.find_all('div', class_='news-card', limit=30)
        
        for card in news_cards:
            try:
                title_tag = card.find('a', class_='title')
                title = title_tag.text.strip() if title_tag else ''
                link = title_tag['href'] if title_tag else ''
                
                snippet_tag = card.find('div', class_='snippet')
                summary = snippet_tag.text.strip() if snippet_tag else ''
                
                if title:
                    articles.append({
                        'source': 'Bing News',
                        'title': title,
                        'link': link,
                        'published': datetime.now(),
                        'summary': summary
                    })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"Bing News error: {e}")
        return []

def get_marketwatch_gold():
    """MarketWatch"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.marketwatch.com/investing/future/gold"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        news_items = soup.find_all('div', class_='article__content', limit=20)
        
        for item in news_items:
            try:
                title_tag = item.find('h3') or item.find('a')
                title = title_tag.text.strip() if title_tag else ''
                link = title_tag['href'] if title_tag and title_tag.get('href') else ''
                
                if not link.startswith('http') and link:
                    link = 'https://www.marketwatch.com' + link
                
                if title:
                    articles.append({
                        'source': 'MarketWatch',
                        'title': title,
                        'link': link,
                        'published': datetime.now(),
                        'summary': ''
                    })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"MarketWatch error: {e}")
        return []

def get_investing_com_gold():
    """Investing.com"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.investing.com/commodities/gold-news"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        news_items = soup.find_all('article', limit=25)
        
        for item in news_items:
            try:
                title_tag = item.find('a')
                title = title_tag.text.strip() if title_tag else ''
                link = title_tag['href'] if title_tag else ''
                
                if link and not link.startswith('http'):
                    link = 'https://www.investing.com' + link
                
                if title:
                    articles.append({
                        'source': 'Investing.com',
                        'title': title,
                        'link': link,
                        'published': datetime.now(),
                        'summary': ''
                    })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"Investing.com error: {e}")
        return []

def get_kitco_news():
    """Kitco"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.kitco.com/news/gold/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        news_items = soup.find_all('div', class_='top-news-item', limit=25)
        
        for item in news_items:
            try:
                title_tag = item.find('a')
                title = title_tag.text.strip() if title_tag else ''
                link = title_tag['href'] if title_tag else ''
                
                if link and not link.startswith('http'):
                    link = 'https://www.kitco.com' + link
                
                if title:
                    articles.append({
                        'source': 'Kitco',
                        'title': title,
                        'link': link,
                        'published': datetime.now(),
                        'summary': ''
                    })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"Kitco error: {e}")
        return []

def get_reuters_gold():
    """Reuters"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.reuters.com/markets/commodities/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        news_items = soup.find_all('a', limit=50)
        
        for item in news_items:
            try:
                title = item.text.strip()
                link = item['href'] if item.get('href') else ''
                
                if 'gold' in title.lower() and link and title:
                    if not link.startswith('http'):
                        link = 'https://www.reuters.com' + link
                    
                    articles.append({
                        'source': 'Reuters',
                        'title': title,
                        'link': link,
                        'published': datetime.now(),
                        'summary': ''
                    })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"Reuters error: {e}")
        return []

def get_cnbc_gold():
    """CNBC"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = "https://www.cnbc.com/commodities/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        news_items = soup.find_all('div', class_='Card-titleContainer', limit=30)
        
        for item in news_items:
            try:
                title_tag = item.find('a')
                title = title_tag.text.strip() if title_tag else ''
                link = title_tag['href'] if title_tag else ''
                
                if 'gold' in title.lower() and title:
                    articles.append({
                        'source': 'CNBC',
                        'title': title,
                        'link': link,
                        'published': datetime.now(),
                        'summary': ''
                    })
            except:
                continue
        
        return articles
    except Exception as e:
        print(f"CNBC error: {e}")
        return []

# -------- COLLECT ALL NEWS --------
def collect_all_news():
    print("="*60)
    print("COLLECTING GOLD NEWS FROM ALL SOURCES")
    print("="*60)
    
    all_articles = []
    
    sources = [
        ("NewsAPI", get_newsapi_gold),
        ("GNews", get_gnews_api),
        ("Yahoo Finance", get_yfinance_news),
        ("Google News RSS", get_google_news),
        ("Bing News", get_bing_news),
        ("Kitco", get_kitco_news),
        ("Investing.com", get_investing_com_gold),
        ("MarketWatch", get_marketwatch_gold),
        ("Reuters", get_reuters_gold),
        ("CNBC", get_cnbc_gold)
    ]
    
    for i, (name, func) in enumerate(sources, 1):
        print(f"\n{i}. {name}...")
        try:
            articles = func()
            print(f"   Found {len(articles)} articles")
            all_articles.extend(articles)
            time.sleep(2)
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n✓ Total articles collected: {len(all_articles)}")
    return all_articles

# -------- FILTER GOLD-RELEVANT ARTICLES --------
def filter_gold_articles(articles, threshold=0.25):
    print("\n" + "="*60)
    print("FILTERING GOLD-RELEVANT ARTICLES")
    print("="*60)
    
    relevant_articles = []
    
    for article in articles:
        text = f"{article['title']} {article.get('summary', '')}"
        
        if not text.strip() or len(text) < 10:
            continue
        
        try:
            article_embedding = model.encode(text)
            similarity = cosine_similarity([gold_embedding], [article_embedding])[0][0]
            
            if similarity >= threshold:
                article['gold_relevance_score'] = float(similarity)
                relevant_articles.append(article)
        except:
            continue
    
    relevant_articles.sort(key=lambda x: x['gold_relevance_score'], reverse=True)
    
    print(f"✓ Filtered to {len(relevant_articles)} gold-relevant articles")
    return relevant_articles

# -------- CLUSTER DUPLICATE NEWS (FIXED) --------
def cluster_duplicate_news(articles, similarity_threshold=0.85):
    """
    Groups similar articles using DBSCAN clustering
    Returns one representative article per cluster
    """
    print("\n" + "="*60)
    print("DEDUPLICATING NEWS (CLUSTERING)")
    print("="*60)
    print(f"Before deduplication: {len(articles)} articles")
    
    if len(articles) < 2:
        print("Not enough articles to cluster")
        return articles
    
    try:
        # Create embeddings for all article titles + summaries
        texts = [f"{article['title']} {article.get('summary', '')}" for article in articles]
        embeddings = model.encode(texts)
        
        # Calculate pairwise similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert to distance and clip to ensure non-negative
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, None)  # FIX: Remove negative values
        
        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=1 - similarity_threshold,
            min_samples=1,
            metric='precomputed'
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        
        # Keep one article per cluster
        unique_articles = []
        cluster_ids = set(clusters)
        
        for cluster_id in cluster_ids:
            cluster_articles = [
                articles[i] for i in range(len(articles)) 
                if clusters[i] == cluster_id
            ]
            
            best_article = max(cluster_articles, key=lambda x: x.get('gold_relevance_score', 0))
            
            best_article['duplicate_count'] = len(cluster_articles)
            best_article['cluster_id'] = int(cluster_id)
            best_article['also_reported_by'] = [a['source'] for a in cluster_articles if a != best_article]
            
            unique_articles.append(best_article)
        
        print(f"After deduplication: {len(unique_articles)} unique stories")
        print(f"Removed {len(articles) - len(unique_articles)} duplicates")
        
        multi_source = [a for a in unique_articles if a['duplicate_count'] > 3]
        if multi_source:
            print(f"\nTop duplicated stories:")
            for i, article in enumerate(multi_source[:3], 1):
                print(f"  {i}. '{article['title'][:60]}...' ({article['duplicate_count']} sources)")
        
        return unique_articles
        
    except Exception as e:
        print(f"Clustering error: {e}")
        print("Returning articles without deduplication")
        for article in articles:
            article['duplicate_count'] = 1
            article['cluster_id'] = -1
            article['also_reported_by'] = []
        return articles

# -------- SENTIMENT ANALYSIS --------
def analyze_sentiment(articles):
    print("\n" + "="*60)
    print("ANALYZING SENTIMENT")
    print("="*60)
    
    bullish_keywords = [
        'rally', 'surge', 'gain', 'rise', 'climb', 'jump', 'soar', 'breakout',
        'bullish', 'support', 'strength', 'buying', 'safe haven', 'record high',
        'inflation', 'rate cut', 'weaker dollar', 'tensions', 'boost', 'higher', 'uptrend'
    ]
    
    bearish_keywords = [
        'fall', 'drop', 'decline', 'plunge', 'tumble', 'sell-off', 'selloff',
        'bearish', 'resistance', 'weakness', 'selling', 'profit taking',
        'rate hike', 'stronger dollar', 'lower', 'correction', 'pressure', 'slump', 'downtrend'
    ]
    
    for article in articles:
        text = f"{article['title']} {article.get('summary', '')}".lower()
        
        bullish_count = sum(1 for kw in bullish_keywords if kw in text)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text)
        
        if bullish_count > bearish_count and bullish_count >= 1:
            article['sentiment'] = 'BULLISH'
            article['sentiment_score'] = bullish_count - bearish_count
        elif bearish_count > bullish_count and bearish_count >= 1:
            article['sentiment'] = 'BEARISH'
            article['sentiment_score'] = bearish_count - bullish_count
        else:
            article['sentiment'] = 'NEUTRAL'
            article['sentiment_score'] = 0
    
    bullish = sum(1 for a in articles if a['sentiment'] == 'BULLISH')
    bearish = sum(1 for a in articles if a['sentiment'] == 'BEARISH')
    neutral = sum(1 for a in articles if a['sentiment'] == 'NEUTRAL')
    
    print(f"✓ BULLISH: {bullish} | BEARISH: {bearish} | NEUTRAL: {neutral}")
    
    return articles

# -------- MAIN --------
def main():
    print("\n🥇 GOLD NEWS ANALYZER (WITH DEDUPLICATION)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Collect
    articles = collect_all_news()
    
    if not articles:
        print("\n❌ No articles collected.")
        return
    
    # Step 2: Filter gold-relevant
    gold_articles = filter_gold_articles(articles, threshold=0.25)
    
    if not gold_articles:
        print("\n❌ No gold-relevant articles found.")
        return
    
    # Step 3: Cluster duplicates
    deduplicated_articles = cluster_duplicate_news(gold_articles, similarity_threshold=0.85)
    
    # Step 4: Analyze sentiment
    analyzed_articles = analyze_sentiment(deduplicated_articles)
    
    # Step 5: Remove neutrals
    filtered_articles = [a for a in analyzed_articles if a['sentiment'] != 'NEUTRAL']
    
    print(f"\n✓ Final filtered: {len(filtered_articles)} (non-neutral)")
    
    # Step 6: Display
    print("\n" + "="*60)
    print("TOP URGENT GOLD NEWS")
    print("="*60)
    
    for i, article in enumerate(filtered_articles[:20], 1):
        icon = "📈" if article['sentiment'] == 'BULLISH' else "📉"
        dup_info = f" [{article['duplicate_count']} sources]" if article.get('duplicate_count', 1) > 1 else ""
        print(f"\n{i}. {icon} [{article['sentiment']}]{dup_info} Relevance: {article['gold_relevance_score']:.2f}")
        print(f"   {article['title']}")
        print(f"   {article['source']}")
        if article.get('also_reported_by'):
            print(f"   Also: {', '.join(article['also_reported_by'][:3])}")
    
    # Step 7: Save
    output_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_collected': len(articles),
        'gold_relevant': len(gold_articles),
        'after_deduplication': len(deduplicated_articles),
        'duplicates_removed': len(gold_articles) - len(deduplicated_articles),
        'final_filtered': len(filtered_articles),
        'sentiment_summary': {
            'bullish': sum(1 for a in filtered_articles if a['sentiment'] == 'BULLISH'),
            'bearish': sum(1 for a in filtered_articles if a['sentiment'] == 'BEARISH')
        },
        'articles': filtered_articles
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✓ Saved to: {OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()