import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import re

# Load environment variables from key.env
load_dotenv("key.env")

class GoldTradingBot:
    def __init__(self):
        # Initialize API clients using environment variables
        self.newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        
        # Metric configuration: Each metric is defined as [weight (percentage), frequency]
        # Total weight should sum to 100%
        self.metrics = {
            'real_rates': [25, 'daily'],       # Triggered if real rates < -0.5
            'fed_policy': [20, 'monthly'],     # Triggered if fed policy == 'dovish'
            'cpi': [20, 'monthly'],            # Triggered if CPI > 3.5
            'geo_risk': [15, 'daily'],         # Triggered if geopolitical risk score >= 7
            'etf_flows': [10, 'weekly'],       # Triggered if ETF flows > 20
            'cb_purchases': [5, 'quarterly'],   # Triggered if central bank purchases > 200
            'dxy': [5, 'daily']                # Triggered if DXY < 100
        }

    def fetch_data(self):
        """Fetch data from all required APIs with fallbacks."""
        today = datetime.today().date()
        data = {}

        # Fetch various economic indicators
        data['real_rates'] = self._fetch_fred_data('DFII10', today)
        data['fed_policy'] = self._fetch_fed_policy(today)
        data['cpi'] = self._fetch_bls_cpi(today)
        data['geo_risk'] = self._fetch_geopolitical_risk(today)
        data['etf_flows'] = self._fetch_etf_flows(today)
        data['cb_purchases'] = self._fetch_cb_purchases(today)
        data['dxy'] = self._fetch_yfinance('DX-Y.NYB', today)

        return data

    def _fetch_fred_data(self, series_id, today):
        """Fetch FRED data with error handling."""
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': os.getenv("FRED_API_KEY"),
            'file_type': 'json'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'observations' not in data or not data['observations']:
                return {'value': None, 'date': today}
            latest_value = float(data['observations'][-1]['value'])
            return {'value': latest_value, 'date': today}
        except Exception as e:
            print(f"Error fetching FRED data: {str(e)}")
            return {'value': None, 'date': today}

    def _fetch_fed_policy(self, today):
        """Mock function for Fed policy data (to be replaced with real data source)."""
        return {'value': 'dovish', 'date': today - timedelta(days=40)}

    def _fetch_bls_cpi(self, today):
        """Mock function for CPI data (to be replaced with real data source)."""
        return {'value': 4.0, 'date': today - timedelta(days=30)}

    def _fetch_etf_flows(self, today):
        """Mock function for ETF flows data (to be replaced with real data source)."""
        return {'value': 22, 'date': today}

    def _fetch_cb_purchases(self, today):
        """Mock function for central bank purchases (to be replaced with real data source)."""
        return {'value': 250, 'date': today - timedelta(days=90)}

    def _fetch_yfinance(self, ticker, today):
        """Fetch Yahoo Finance data with error handling."""
        try:
            data = yf.Ticker(ticker).history(period='1d')
            return {'value': data['Close'].iloc[-1], 'date': today}
        except Exception as e:
            print(f"Error fetching {ticker} data: {str(e)}")
            return {'value': None, 'date': today}

    def _fetch_geopolitical_risk(self, today):
        """Fetch geopolitical risk based on relevant news articles with improved filtering."""
        keywords = [
            "war", "invasion", "attack", "military operation", "airstrike", "bombing", "conflict",
            "battle", "frontline", "insurgency", "shelling", "missile strike", "drone strike",
            "crisis", "tensions", "sanctions", "border dispute", "uprising", "skirmish", "coup",
            "armed clashes", "retaliation", "escalation", "ceasefire violation", "diplomatic breakdown",
            "terror attack", "extremism", "political unrest", "nationalist movement"
        ]
        query = " OR ".join(keywords)
        try:
            articles = self.newsapi.get_everything(
                q=query,
                from_param=(today - timedelta(days=1)).isoformat(),
                to=today.isoformat(),
                language='en',
                sort_by='relevancy'
            )
            if not articles or 'articles' not in articles or not articles['articles']:
                return {'value': 0, 'date': today}
            
            valid_articles = 0
            party_pattern = re.compile(
                r"\b(Russia|Ukraine|Israel|Palestine|Hamas|USA|China|Taiwan|Iran|NATO|India|Pakistan|North Korea|South Korea|Syria|Yemen|Saudi Arabia|Turkey)\b",
                re.IGNORECASE
            )
            for article in articles['articles']:
                content = f"{article.get('title', '')} {article.get('description', '')}"
                found_parties = set(party_pattern.findall(content))
                if len(found_parties) >= 2:
                    valid_articles += 1
            severity = min(valid_articles * 2, 10)
            return {'value': severity, 'date': today}
        except Exception as e:
            print(f"Error fetching geopolitical risk: {str(e)}")
            return {'value': 0, 'date': today}

    def calculate_signal(self):
        """
        Generate trading signal based on fetched data using weighted metrics.
        Returns:
            signal (str): 'BUY', 'SELL', or 'HOLD'
            total_score (int): The overall weighted score (in percentage).
            key_drivers (list): List of metric names that triggered.
        """
        data = self.fetch_data()
        total_score = 0
        key_drivers = []
        
        for metric, config in self.metrics.items():
            weight, frequency = config
            value = data.get(metric, {}).get('value')
            if self._is_triggered(metric, value):
                total_score += weight
                key_drivers.append(metric)
        
        if total_score >= 60:
            signal = "BUY"
        elif total_score <= 40:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return signal, total_score, key_drivers

    def _is_triggered(self, metric, value):
        """Condition checks for each metric."""
        conditions = {
            'real_rates': lambda v: v is not None and v < -0.5,
            'fed_policy': lambda v: v is not None and v.lower() == 'dovish',
            'cpi': lambda v: v is not None and v > 3.5,
            'geo_risk': lambda v: v is not None and v >= 7,
            'etf_flows': lambda v: v is not None and v > 20,
            'cb_purchases': lambda v: v is not None and v > 200,
            'dxy': lambda v: v is not None and v < 100,
        }
        return conditions[metric](value)

    def format_output(self, signal, total_score, key_drivers):
        """Generate formatted output including overall rate and key drivers."""
        output = [
            "\n=== GOLD TRADING SIGNAL ===",
            f"Signal: {signal}",
            f"Overall Rate: {total_score}%",
            f"Key Drivers: {', '.join(key_drivers) if key_drivers else 'None'}",
            "=" * 30
        ]
        return "\n".join(output)

if __name__ == "__main__":
    bot = GoldTradingBot()
    signal, total_score, key_drivers = bot.calculate_signal()
    print(bot.format_output(signal, total_score, key_drivers))
