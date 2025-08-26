# disease_outbreak_prediction/src/data_collection/social_media_collector.py

import pandas as pd
import numpy as np
import tweepy
import praw
import requests
from datetime import datetime, timedelta
import json
import time
from textblob import TextBlob
import re


class SocialMediaCollector:
    """
    Collects data from various social media platforms for disease outbreak prediction
    """

    def __init__(self):
        self.twitter_data = []
        self.reddit_data = []
        self.health_keywords = [
            'fever', 'cough', 'flu', 'sick', 'illness', 'outbreak', 'epidemic',
            'pandemic', 'symptoms', 'headache', 'fatigue', 'sore throat',
            'nausea', 'vomiting', 'diarrhea', 'respiratory', 'covid', 'virus'
        ]

    def setup_twitter_api(self, bearer_token):
        """Setup Twitter API v2 client"""
        try:
            self.twitter_client = tweepy.Client(bearer_token=bearer_token)
            return True
        except Exception as e:
            print(f"Twitter API setup failed: {e}")
            return False

    def setup_reddit_api(self, client_id, client_secret, user_agent):
        """Setup Reddit API client"""
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            return True
        except Exception as e:
            print(f"Reddit API setup failed: {e}")
            return False

    def collect_twitter_data(self, query, max_results=100, days_back=7):
        """
        Collect tweets related to health symptoms and diseases
        """
        try:
            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            # Search for tweets
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'public_metrics',
                              'context_annotations', 'geo'],
                user_fields=['location', 'verified'],
                expansions=['author_id'],
                start_time=start_time,
                end_time=end_time,
                max_results=min(max_results, 100)
            ).flatten(limit=max_results)

            tweet_data = []
            for tweet in tweets:
                tweet_info = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'quote_count': tweet.public_metrics['quote_count'],
                    'author_id': tweet.author_id,
                    'source': 'twitter'
                }

                # Add location if available
                if hasattr(tweet, 'geo') and tweet.geo:
                    tweet_info['location'] = tweet.geo

                tweet_data.append(tweet_info)

            self.twitter_data.extend(tweet_data)
            return tweet_data

        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
            return []

    def collect_reddit_data(self, subreddits=['medical', 'AskDocs', 'HealthAnxiety'], limit=100):
        """
        Collect Reddit posts from health-related subreddits
        """
        try:
            reddit_posts = []

            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Get recent posts
                for post in subreddit.new(limit=limit):
                    # Filter for health-related content
                    if any(keyword in post.title.lower() or
                           keyword in post.selftext.lower()
                           for keyword in self.health_keywords):

                        post_data = {
                            'id': post.id,
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'subreddit': subreddit_name,
                            'author': str(post.author) if post.author else 'deleted',
                            'source': 'reddit'
                        }
                        reddit_posts.append(post_data)

            self.reddit_data.extend(reddit_posts)
            return reddit_posts

        except Exception as e:
            print(f"Error collecting Reddit data: {e}")
            return []

    def get_google_trends_data(self, keywords, timeframe='today 1-m', geo=''):
        """
        Simulate Google Trends data collection (requires pytrends library)
        """
        # Note: Install pytrends with: pip install pytrends
        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(
                keywords, cat=0, timeframe=timeframe, geo=geo)

            # Get interest over time
            interest_over_time = pytrends.interest_over_time()

            # Get interest by region
            interest_by_region = pytrends.interest_by_region(
                resolution='COUNTRY')

            return {
                'interest_over_time': interest_over_time,
                'interest_by_region': interest_by_region
            }

        except ImportError:
            print("pytrends not installed. Install with: pip install pytrends")
            return None
        except Exception as e:
            print(f"Error collecting Google Trends data: {e}")
            return None

    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Convert to lowercase
        text = text.lower()

        return text

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0, 'subjectivity': 0}

    def extract_health_symptoms(self, text):
        """
        Extract health-related symptoms from text
        """
        symptoms_found = []
        text_lower = text.lower()

        for keyword in self.health_keywords:
            if keyword in text_lower:
                symptoms_found.append(keyword)

        return symptoms_found

    def save_data(self, filename='social_media_data.csv'):
        """
        Save collected data to CSV file
        """
        all_data = []

        # Process Twitter data
        for tweet in self.twitter_data:
            processed = {
                'id': tweet['id'],
                'platform': 'twitter',
                'text': self.preprocess_text(tweet['text']),
                'raw_text': tweet['text'],
                'created_at': tweet['created_at'],
                'engagement_score': tweet['like_count'] + tweet['retweet_count'],
                'author': tweet['author_id']
            }

            # Add sentiment analysis
            sentiment = self.analyze_sentiment(processed['text'])
            processed.update(sentiment)

            # Extract symptoms
            processed['symptoms'] = self.extract_health_symptoms(
                processed['text'])

            all_data.append(processed)

        # Process Reddit data
        for post in self.reddit_data:
            text_content = f"{post['title']} {post['text']}"
            processed = {
                'id': post['id'],
                'platform': 'reddit',
                'text': self.preprocess_text(text_content),
                'raw_text': text_content,
                'created_at': post['created_utc'],
                'engagement_score': post['score'],
                'author': post['author']
            }

            # Add sentiment analysis
            sentiment = self.analyze_sentiment(processed['text'])
            processed.update(sentiment)

            # Extract symptoms
            processed['symptoms'] = self.extract_health_symptoms(
                processed['text'])

            all_data.append(processed)

        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

        return df


# Example usage and testing
if __name__ == "__main__":
    # Initialize collector
    collector = SocialMediaCollector()

    # Example: Collect sample data (you'll need to add your API keys)
    # collector.setup_twitter_api("your_bearer_token")
    # collector.setup_reddit_api("client_id", "client_secret", "user_agent")

    # For demonstration, create some sample data
    sample_tweets = [
        {
            'id': '123456789',
            'text': 'Feeling sick with fever and cough. Is this flu season starting early?',
            'created_at': datetime.now(),
            'retweet_count': 5,
            'like_count': 12,
            'reply_count': 3,
            'quote_count': 1,
            'author_id': 'user123',
            'source': 'twitter'
        }
    ]

    collector.twitter_data = sample_tweets

    # Process and save data
    df = collector.save_data('sample_social_media_data.csv')
    print(f"Processed {len(df)} social media posts")
    print(df.head())
