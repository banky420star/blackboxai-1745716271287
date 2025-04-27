"""
Sentiment analysis module using FinBERT for financial news
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import numpy as np
from loguru import logger

class SentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT for financial news
    """
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name: Name of the pre-trained FinBERT model
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Sentiment analyzer initialized using {model_name}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise

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

    def get_sentiment_features(self, news_data: List[Dict[str, str]]) -> np.ndarray:
        """
        Extract sentiment features from news data
        
        Args:
            news_data: List of dictionaries containing news items with 'title' and 'text' keys
            
        Returns:
            Numpy array of sentiment features
        """
        try:
            # Combine titles and texts for analysis
            texts = [
                f"{item.get('title', '')} {item.get('text', '')}"
                for item in news_data
            ]
            
            # Get sentiment analysis results
            sentiments = self.analyze_batch(texts)
            
            # Extract features
            features = np.array([
                [
                    s['scores']['positive'],
                    s['scores']['negative'],
                    s['scores']['neutral'],
                    s['compound_score']
                ]
                for s in sentiments
            ])
            
            # Aggregate features
            if len(features) > 0:
                mean_features = np.mean(features, axis=0)
                std_features = np.std(features, axis=0)
                max_features = np.max(features, axis=0)
                min_features = np.min(features, axis=0)
                
                return np.concatenate([
                    mean_features,
                    std_features,
                    max_features,
                    min_features
                ])
            else:
                # Return zero features if no news data
                return np.zeros(16)  # 4 features * 4 aggregations
            
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {str(e)}")
            return np.zeros(16)
