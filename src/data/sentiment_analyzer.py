"""
Sentiment analysis module using FinBERT for financial news
"""
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from loguru import logger
import requests
from bs4 import BeautifulSoup
import time

class SentimentAnalyzer:
    """
    Handles sentiment analysis of financial news using FinBERT
    """
    
    def __init__(self, config: dict):
        """
        Initialize the sentiment analyzer
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.setup_logging()
        self.setup_model()
        
    def setup_logging(self):
        """Configure logging"""
        logger.add(
            "logs/sentiment_analyzer_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    def setup_model(self):
        """Initialize FinBERT model"""
        try:
            model_name = "yiyanghkust/finbert-tone"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"FinBERT model loaded successfully. Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise
            
    def get_news_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        source: str = 'benzinga'
    ) -> List[Dict]:
        """
        Fetch financial news articles
        
        Args:
            symbol: Stock/crypto symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: News source ('benzinga', 'reuters', etc.)
            
        Returns:
            List of dictionaries containing news articles
        """
        try:
            if source == 'benzinga':
                if not self.config.get('benzinga_api_key'):
                    raise ValueError("Benzinga API key not provided")
                    
                headers = {
                    'Authorization': f"Token {self.config['benzinga_api_key']}"
                }
                
                params = {
                    'symbols': symbol,
                    'from': start_date,
                    'to': end_date,
                    'pageSize': 100
                }
                
                response = requests.get(
                    'https://api.benzinga.com/api/v2/news',
                    headers=headers,
                    params=params
                )
                
                if response.status_code != 200:
                    logger.error(f"Error fetching news: {response.status_code}")
                    return []
                    
                articles = response.json()
                
                # Process articles
                processed_articles = []
                for article in articles:
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'date': article.get('created', ''),
                        'source': 'benzinga'
                    })
                    
                logger.info(f"Fetched {len(processed_articles)} articles from Benzinga")
                return processed_articles
                
            else:
                logger.warning(f"News source {source} not supported")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return []
            
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get sentiment prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                
            # Convert to numpy and get scores
            scores = probabilities.cpu().numpy()[0]
            
            # Map scores to sentiments (FinBERT specific)
            sentiment_scores = {
                "positive": float(scores[0]),
                "negative": float(scores[1]),
                "neutral": float(scores[2])
            }
            
            # Get dominant sentiment
            dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            
            return {
                "scores": sentiment_scores,
                "dominant_sentiment": dominant_sentiment,
                "compound_score": float(scores[0] - scores[1])  # positive - negative
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "dominant_sentiment": "neutral",
                "compound_score": 0.0
            }
            
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries containing sentiment scores
        """
        try:
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get sentiment predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Convert to numpy and process each result
            scores_batch = probabilities.cpu().numpy()
            results = []
            
            for scores in scores_batch:
                sentiment_scores = {
                    "positive": float(scores[0]),
                    "negative": float(scores[1]),
                    "neutral": float(scores[2])
                }
                
                dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
                
                results.append({
                    "scores": sentiment_scores,
                    "dominant_sentiment": dominant_sentiment,
                    "compound_score": float(scores[0] - scores[1])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {str(e)}")
            return [
                {
                    "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                    "dominant_sentiment": "neutral",
                    "compound_score": 0.0
                }
                for _ in texts
            ]
            
    def get_sentiment_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        source: str = 'benzinga'
    ) -> pd.DataFrame:
        """
        Get sentiment features for a symbol over a time period
        
        Args:
            symbol: Stock/crypto symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: News source ('benzinga', 'reuters', etc.)
            
        Returns:
            DataFrame with sentiment features indexed by date
        """
        try:
            # Fetch news articles
            articles = self.get_news_data(symbol, start_date, end_date, source)
            
            if not articles:
                logger.warning(f"No articles found for {symbol}")
                return pd.DataFrame()
            
            # Group articles by date
            articles_by_date = {}
            for article in articles:
                date = article['date'].split('T')[0]  # Get date part only
                if date not in articles_by_date:
                    articles_by_date[date] = []
                articles_by_date[date].append(article)
            
            # Analyze sentiment for each date
            sentiment_data = []
            for date, daily_articles in articles_by_date.items():
                # Combine title and content for each article
                texts = [
                    f"{article['title']} {article['content']}"
                    for article in daily_articles
                ]
                
                # Get sentiment scores for all articles
                sentiments = self.analyze_batch(texts)
                
                # Calculate aggregate features
                scores = np.array([
                    [
                        s['scores']['positive'],
                        s['scores']['negative'],
                        s['scores']['neutral'],
                        s['compound_score']
                    ]
                    for s in sentiments
                ])
                
                sentiment_data.append({
                    'date': date,
                    'positive_mean': scores[:, 0].mean(),
                    'negative_mean': scores[:, 1].mean(),
                    'neutral_mean': scores[:, 2].mean(),
                    'compound_mean': scores[:, 3].mean(),
                    'positive_std': scores[:, 0].std(),
                    'negative_st
