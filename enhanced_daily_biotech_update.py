#!/usr/bin/env python3
"""
enhanced_daily_biotech_update.py
=================================

Enhanced daily biotech update script with comprehensive features:
- Local SQLite database for data persistence and deduplication
- Advanced filtering and categorization
- Multiple output formats (HTML email, JSON export, dashboard)
- Robust error handling and retry mechanisms
- Analytics and reporting capabilities
- Configuration management
- Comprehensive logging with different log levels

Dependencies:
    pip install feedparser requests beautifulsoup4 python-dateutil pyyaml
    pip install pandas matplotlib seaborn python-dotenv
"""

import json
import logging
import os
import smtplib
import sqlite3
import hashlib
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urljoin
import re
import sys

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dateutil import tz

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not installed. Install with: pip install python-dotenv")

# Set up matplotlib for headless operation
plt.switch_backend('Agg')

# ----------------------------------------------------------------------
# Configuration and Data Models

@dataclass
class Article:
    """Data model for a news article"""
    title: str
    link: str
    published: datetime
    summary: str
    content: str
    source_name: str
    category: str
    content_hash: str
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    word_count: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(tz=tz.UTC)
        if not self.content_hash:
            self.content_hash = self._generate_hash()
        if not self.word_count:
            self.word_count = len(self.title.split()) + len(self.summary.split())
    
    def _generate_hash(self) -> str:
        """Generate unique hash for deduplication"""
        content = f"{self.title}{self.link}{self.summary}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

@dataclass
class DataSource:
    """Data model for a data source"""
    name: str
    category: str
    source_type: str  # 'rss', 'html', 'api'
    url: str
    enabled: bool = True
    last_fetched: Optional[datetime] = None
    fetch_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    keywords: List[str] = None
    custom_selectors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.custom_selectors is None:
            self.custom_selectors = {}

class WebhookNotifier:
    """Send notifications via webhook (Slack, Discord, etc.)"""
    
    def __init__(self, webhook_url: str, webhook_type: str = 'slack'):
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type.lower()
        self.channel = None
        self.username = "Biotech Update Bot"
    
    def send_notification(self, title: str, message: str, articles_count: int = 0) -> bool:
        """Send notification via webhook"""
        try:
            if self.webhook_type == 'slack':
                payload = {
                    "text": f"*{title}*",
                    "username": self.username,
                    "attachments": [
                        {
                            "color": "good" if articles_count > 0 else "warning",
                            "text": message,
                            "fields": [
                                {
                                    "title": "Articles Found",
                                    "value": str(articles_count),
                                    "short": True
                                },
                                {
                                    "title": "Timestamp",
                                    "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "short": True
                                }
                            ]
                        }
                    ]
                }
                if self.channel:
                    payload["channel"] = self.channel
                    
            elif self.webhook_type == 'discord':
                payload = {
                    "username": self.username,
                    "embeds": [
                        {
                            "title": title,
                            "description": message,
                            "color": 0x00ff00 if articles_count > 0 else 0xffaa00,
                            "fields": [
                                {
                                    "name": "Articles Found",
                                    "value": str(articles_count),
                                    "inline": True
                                }
                            ],
                            "timestamp": datetime.now().isoformat()
                        }
                    ]
                }
            elif self.webhook_type == 'teams':
                payload = {
                    "@type": "MessageCard",
                    "@context": "https://schema.org/extensions",
                    "summary": title,
                    "themeColor": "00ff00" if articles_count > 0 else "ffaa00",
                    "sections": [
                        {
                            "activityTitle": title,
                            "activitySubtitle": message,
                            "facts": [
                                {
                                    "name": "Articles Found",
                                    "value": str(articles_count)
                                },
                                {
                                    "name": "Timestamp",
                                    "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                            ]
                        }
                    ]
                }
            else:
                # Generic webhook
                payload = {
                    "title": title,
                    "message": message,
                    "articles_count": articles_count,
                    "timestamp": datetime.now().isoformat(),
                    "username": self.username
                }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logging.error(f"Webhook notification failed: {e}")
            return False

class BiotechUpdateManager:
    """Main class for managing the biotech update workflow"""
    
    # Define keyword categories for article classification
    OBESITY_GLP1_KEYWORDS = [
        'glp-1', 'glp1', 'glp 1', 'obesity', 'weight loss', 'weight management',
        'semaglutide', 'tirzepatide', 'ozempic', 'wegovy', 'mounjaro', 'zepbound',
        'liraglutide', 'saxenda', 'victoza', 'dulaglutide', 'trulicity',
        'novo nordisk', 'eli lilly', 'incretins', 'gip', 'gip/glp-1',
        'dual agonist', 'triple agonist', 'retatrutide', 'orforglipron',
        'amg 133', 'cagrisema', 'survodutide', 'mazdutide', 'pemvidutide',
        'ecnoglutide', 'vk2735', 'bi 456906', 'metabolic disease',
        'diabetes', 'type 2 diabetes', 't2d', 'nash', 'nafld', 
        'cardiovascular benefit', 'weight reduction', 'bmi reduction',
        'appetite suppression', 'satiety', 'glp-1 receptor agonist'
    ]
    
    GENE_CELL_THERAPY_KEYWORDS = [
        'gene therapy', 'cell therapy', 'cgt', 'car-t', 'car t', 
        'crispr', 'gene editing', 'aav', 'lentiviral', 'adeno-associated',
        'gene transfer', 'gene correction', 'gene replacement', 'ex vivo',
        'in vivo', 'autologous', 'allogeneic', 'stem cell', 'ipsc',
        'induced pluripotent', 'hematopoietic', 'mesenchymal', 'tcr',
        'tcr-t', 'til', 'tumor infiltrating', 'base editing', 'prime editing',
        'zinc finger', 'talen', 'meganuclease', 'transposon', 'sleeping beauty',
        'piggyback', 'viral vector', 'non-viral delivery', 'lipid nanoparticle',
        'electroporation', 'nucleofection', 'gene silencing', 'rnai', 'sirna',
        'antisense', 'gene augmentation', 'oncolytic virus', 'gene drive'
    ]
    
    ONCOLOGY_KEYWORDS = [
        'oncology', 'cancer', 'tumor', 'carcinoma', 'lymphoma', 
        'leukemia', 'melanoma', 'metastatic', 'adc', 'antibody drug conjugate',
        'checkpoint inhibitor', 'pd-1', 'pd-l1', 'ctla-4', 'immunotherapy',
        'targeted therapy', 'kinase inhibitor', 'monoclonal antibody',
        'bispecific', 'trispecific', 'radioligand', 'radiopharmaceutical',
        'braf', 'kras', 'egfr', 'her2', 'alk', 'ros1', 'met', 'ret',
        'ntrk', 'fgfr', 'vegf', 'bcl-2', 'btk', 'cdk4/6', 'parp',
        'protac', 'degrader', 'payload', 'linker', 'dar', 'solid tumor',
        'hematologic malignancy', 'nsclc', 'sclc', 'breast cancer',
        'prostate cancer', 'colorectal', 'pancreatic', 'ovarian',
        'glioblastoma', 'multiple myeloma', 'aml', 'all', 'cll', 'dlbcl'
    ]
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the manager with configuration"""
        # Handle Windows path and ensure we're in the correct directory
        script_dir = Path(__file__).parent.absolute()
        target_dir = Path(r"C:\Users\cathe\OneDrive\文档\Risk\Project folder\biotech daily digest")
        
        # Change to target directory if it exists, otherwise use script directory
        if target_dir.exists():
            os.chdir(target_dir)
            print(f"Working directory set to: {target_dir}")
        else:
            print(f"Target directory not found, using: {script_dir}")
            os.chdir(script_dir)
        
        # Load environment variables from .env file in current directory
        env_path = Path(".env")
        if env_path.exists() and DOTENV_AVAILABLE:
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                print(f"Loaded environment variables from: {env_path.absolute()}")
            except Exception as e:
                print(f"Error loading .env file: {e}")
        else:
            print(f".env file not found or dotenv not available")
        
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'biotech_updates.db')
        self.lookback_days = self.config.get('general', {}).get('lookback_days', 3)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize database
        self._init_database()
        
        # Load data sources
        self.sources = self._load_data_sources()
        
        # Set up timezone
        self.tz = tz.gettz(self.config.get('general', {}).get('timezone', 'America/New_York'))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("BiotechUpdateManager initialized")
        self.logger.info(f"Working directory: {Path.cwd()}")
        self.logger.info(f"Database path: {Path(self.db_path).absolute()}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file and .env variables"""
        default_config = {
            'general': {
                'lookback_days': 3,
                'timezone': 'America/New_York',
                'max_articles_per_source': 50,
                'retry_attempts': 3,
                'retry_delay': 1.0
            },
            'database': {
                'path': 'biotech_updates.db',
                'backup_frequency': 7
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/biotech_update.log',
                'max_size': '10MB',
                'backup_count': 5
            },
            'email': {
                'enabled': self._get_env_bool('EMAIL_ENABLED', True),
                'smtp_host': os.getenv('SMTP_HOST', ''),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'smtp_user': os.getenv('SMTP_USER', ''),
                'smtp_password': os.getenv('SMTP_PASSWORD', ''),
                'sender': os.getenv('REPORT_SENDER', ''),
                'recipients': self._parse_recipients(os.getenv('REPORT_RECIPIENTS', '')),
                'subject': os.getenv('REPORT_SUBJECT', 'Daily Biotech Update - {date}')
            },
            'webhook': {
                'enabled': self._get_env_bool('WEBHOOK_ENABLED', False),
                'url': os.getenv('WEBHOOK_URL', ''),
                'type': os.getenv('WEBHOOK_TYPE', 'slack'),
                'channel': os.getenv('WEBHOOK_CHANNEL', ''),
                'username': os.getenv('WEBHOOK_USERNAME', 'Biotech Update Bot')
            },
            'filtering': {
                'min_word_count': int(os.getenv('MIN_WORD_COUNT', '10')),
                'excluded_keywords': self._parse_list(os.getenv('EXCLUDED_KEYWORDS', 'spam,advertisement')),
                'required_keywords': self._parse_list(os.getenv('REQUIRED_KEYWORDS', '')),
                'relevance_threshold': float(os.getenv('RELEVANCE_THRESHOLD', '0.0'))
            },
            'features': {
                'enable_sentiment_analysis': self._get_env_bool('ENABLE_SENTIMENT_ANALYSIS', False),
                'enable_webhook_notifications': self._get_env_bool('ENABLE_WEBHOOK_NOTIFICATIONS', False),
                'enable_dashboard': self._get_env_bool('ENABLE_DASHBOARD', True),
                'enable_analytics': self._get_env_bool('ENABLE_ANALYTICS', True)
            },
            'sources_file': os.getenv('SOURCES_FILE', 'data_sources.json')
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            # Merge configurations
            config = self._merge_configs(default_config, user_config)
        else:
            config = default_config
            
        return config

    def _get_env_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse comma-separated list of email recipients"""
        if not recipients_str:
            return []
        return [email.strip() for email in recipients_str.split(',') if email.strip()]

    def _parse_list(self, list_str: str) -> List[str]:
        """Parse comma-separated list from string"""
        if not list_str:
            return []
        return [item.strip() for item in list_str.split(',') if item.strip()]

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        merged = default.copy()
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _setup_logging(self):
        """Configure logging with rotation and multiple levels"""
        log_config = self.config.get('logging', {})
        log_dir = Path(log_config.get('file', 'logs/biotech_update.log')).parent
        log_dir.mkdir(exist_ok=True)
        
        from logging.handlers import RotatingFileHandler
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                RotatingFileHandler(
                    log_config.get('file', 'logs/biotech_update.log'),
                    maxBytes=self._parse_size(log_config.get('max_size', '10MB')),
                    backupCount=log_config.get('backup_count', 5),
                    encoding='utf-8'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def _init_database(self):
        """Initialize SQLite database with all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT NOT NULL,
                published TIMESTAMP,
                summary TEXT,
                content TEXT,
                source_name TEXT,
                category TEXT,
                content_hash TEXT UNIQUE,
                sentiment_score REAL DEFAULT 0.0,
                relevance_score REAL DEFAULT 0.0,
                word_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                included_in_report BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Data sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT,
                source_type TEXT,
                url TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                last_fetched TIMESTAMP,
                fetch_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0.0,
                keywords TEXT,
                custom_selectors TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(content_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category)')
        
        conn.commit()
        conn.close()
        
        logging.info(f"Database initialized at {self.db_path}")

    def _load_data_sources(self) -> List[DataSource]:
        """Load data sources from JSON file"""
        sources_file = self.config.get('sources_file', 'data_sources.json')
        sources = []
        
        if os.path.exists(sources_file):
            try:
                with open(sources_file, 'r', encoding='utf-8') as f:
                    sources_data = json.load(f)
                
                for src_data in sources_data:
                    source = DataSource(
                        name=src_data.get('name', ''),
                        category=src_data.get('category', 'Uncategorized'),
                        source_type=src_data.get('type', 'rss'),
                        url=src_data.get('url', ''),
                        enabled=src_data.get('enabled', True),
                        keywords=src_data.get('keywords', []),
                        custom_selectors=src_data.get('custom_selectors', {})
                    )
                    sources.append(source)
                    
            except Exception as e:
                logging.error(f"Failed to load data sources: {e}")
        
        return sources

    def fetch_all_updates(self) -> Dict[str, List[Article]]:
        """Fetch updates from all enabled sources"""
        updates = {}
        
        for source in self.sources:
            if not source.enabled:
                continue
                
            try:
                if source.source_type.lower() == 'rss':
                    articles = self._fetch_rss_articles(source)
                else:
                    logging.warning(f"Source type {source.source_type} not implemented for {source.name}")
                    continue
                
                # Process articles
                new_articles = self._process_articles(articles, source.category)
                
                if new_articles:
                    updates[source.category] = updates.get(source.category, []) + new_articles
                
                logging.info(f"Fetched {len(new_articles)} new articles from {source.name}")
                
            except Exception as e:
                logging.error(f"Failed to fetch from {source.name}: {e}")
        
        return updates

    def _fetch_rss_articles(self, source: DataSource) -> List[Article]:
        """Fetch articles from RSS feed"""
        try:
            feed = feedparser.parse(source.url)
            articles = []
            
            for entry in feed.entries[:self.config['general']['max_articles_per_source']]:
                article = Article(
                    title=entry.get('title', '').strip(),
                    link=entry.get('link', ''),
                    published=self._parse_entry_date(entry),
                    summary=self._extract_summary(entry),
                    content=self._extract_content(entry),
                    source_name=source.name,
                    category=source.category,
                    content_hash=''  # Will be generated in __post_init__
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logging.error(f"Error fetching RSS from {source.name}: {e}")
            raise

    def _parse_entry_date(self, entry: Dict[str, Any]) -> datetime:
        """Parse publication date from feed entry"""
        date_fields = ['published', 'updated', 'pubDate', 'date']
        
        for field in date_fields:
            date_str = entry.get(field)
            if date_str:
                try:
                    dt = date_parser.parse(date_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=tz.UTC)
                    return dt.astimezone(self.tz)
                except Exception:
                    continue
        
        return datetime.now(self.tz)

    def _extract_summary(self, entry: Dict[str, Any], max_length: int = 300) -> str:
        """Extract and clean summary from entry"""
        summary_fields = ['summary', 'description', 'content']
        
        for field in summary_fields:
            content = entry.get(field, '')
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('value', '')
            
            if content:
                soup = BeautifulSoup(str(content), 'html.parser')
                clean_text = soup.get_text().strip()
                return (clean_text[:max_length] + '…') if len(clean_text) > max_length else clean_text
        
        return ''

    def _extract_content(self, entry: Dict[str, Any]) -> str:
        """Extract full content from entry if available"""
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            content = entry.get(field, '')
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('value', '')
            
            if content:
                soup = BeautifulSoup(str(content), 'html.parser')
                return soup.get_text().strip()
        
        return ''

    def _process_articles(self, articles: List[Article], category: str) -> List[Article]:
        """Filter, deduplicate, and score articles"""
        processed_articles = []
        cutoff_date = datetime.now(self.tz) - timedelta(days=self.lookback_days)
        
        for article in articles:
            # Date filter
            if article.published < cutoff_date:
                continue
            
            # Content filters
            if not self._passes_content_filter(article):
                continue
            
            # Check for duplicates
            if self._is_duplicate(article):
                continue
            
            # Calculate scores
            article.relevance_score = self._calculate_relevance_score(article)
            article.sentiment_score = self._calculate_sentiment_score(article)
            
            # Save to database
            self._save_article_to_db(article)
            
            processed_articles.append(article)
        
        return processed_articles

    def _passes_content_filter(self, article: Article) -> bool:
        """Apply content filtering rules"""
        config = self.config.get('filtering', {})
        text = f"{article.title} {article.summary}".lower()
        
        # Always pass high-priority articles
        if any(term in text for term in self.OBESITY_GLP1_KEYWORDS):
            return True
        if any(term in text for term in self.GENE_CELL_THERAPY_KEYWORDS):
            return True
        
        # Word count filter
        if article.word_count < config.get('min_word_count', 10):
            return False
        
        # Excluded keywords
        excluded = config.get('excluded_keywords', [])
        if any(keyword.lower() in text for keyword in excluded):
            return False
        
        return True

    def _is_duplicate(self, article: Article) -> bool:
        """Check if article already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM articles WHERE content_hash = ?', (article.content_hash,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None

    def _calculate_relevance_score(self, article: Article) -> float:
        """Calculate relevance score based on keywords and content"""
        text = f"{article.title} {article.summary}".lower()
        score = 0.0
        
        # High-priority keywords (weight: 2.0 each)
        for keyword in self.OBESITY_GLP1_KEYWORDS:
            if keyword in text:
                score += 2.0
        
        # Medium-priority keywords (weight: 1.5 each)
        for keyword in self.GENE_CELL_THERAPY_KEYWORDS:
            if keyword in text:
                score += 1.5
        
        # Standard keywords (weight: 1.0 each)
        for keyword in self.ONCOLOGY_KEYWORDS:
            if keyword in text:
                score += 1.0
        
        # Normalize by content length and cap at 10.0
        normalized_score = score / max(article.word_count / 100, 1)
        final_score = min(normalized_score, 10.0)
        
        # Ensure high-priority articles get minimum scores
        if any(keyword in text for keyword in ['glp-1', 'glp1', 'obesity']):
            final_score = max(final_score, 3.0)
        if any(keyword in text for keyword in ['gene therapy', 'cell therapy', 'car-t']):
            final_score = max(final_score, 2.5)
        
        return final_score

    def _calculate_sentiment_score(self, article: Article) -> float:
        """Calculate basic sentiment score"""
        positive_words = ['approval', 'success', 'breakthrough', 'positive', 'advance']
        negative_words = ['failure', 'decline', 'reject', 'negative', 'concern']
        
        text = f"{article.title} {article.summary}".lower()
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)

    def _save_article_to_db(self, article: Article):
        """Save article to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO articles 
                (title, link, published, summary, content, source_name, category, 
                 content_hash, sentiment_score, relevance_score, word_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.title,
                article.link,
                article.published,
                article.summary,
                article.content,
                article.source_name,
                article.category,
                article.content_hash,
                article.sentiment_score,
                article.relevance_score,
                article.word_count,
                article.created_at
            ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error saving article to database: {e}")
        finally:
            conn.close()

    def _categorize_article(self, article: Article) -> str:
        """Categorize article into Obesity/GLP-1, Cell/Gene, or General"""
        text = f"{article.title} {article.summary}".lower()
        
        if any(keyword in text for keyword in self.OBESITY_GLP1_KEYWORDS):
            return "Obesity/GLP-1"
        if any(keyword in text for keyword in self.GENE_CELL_THERAPY_KEYWORDS):
            return "Cell/Gene Therapy"
        
        return "General"

    def generate_html_report(self, updates: Dict[str, List[Article]]) -> str:
        """Generate HTML report"""
        report_date = datetime.now(self.tz)
        date_str = report_date.strftime("%B %d, %Y")
        
        total_articles = sum(len(articles) for articles in updates.values())
        industry_articles = len(updates.get('Industry', []))
        company_articles = total_articles - industry_articles
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8"/>',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>',
            f'<title>Daily Biotech Update — {date_str}</title>',
            self._get_email_styles(),
            '</head>',
            '<body>',
            '<div class="wrapper">',
            '<div class="container">',
            '<div class="header">',
            '<h1>Daily Biotech Update</h1>',
            f'<p>{date_str} • Window: last {self.lookback_days} days • {total_articles} articles</p>',
            '</div>',
            '<div class="content">'
        ]
        
        # Industry Update Section
        html_parts.extend(self._generate_industry_section(updates.get('Industry', [])))
        html_parts.append('<div class="sep"></div>')
        
        # Company Updates Section
        html_parts.extend(self._generate_company_section(updates))
        
        html_parts.extend([
            '</div>',
            f'<div class="footer">Generated automatically at {report_date.strftime("%Y-%m-%d %H:%M %Z")}</div>',
            '</div>',
            '</div>',
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)

    def _get_email_styles(self) -> str:
        """Get CSS styles for the email"""
        return '''
<style>
  body { margin:0; padding:0; background:#f5f7fb; -webkit-font-smoothing: antialiased; }
  .wrapper { width:100%; background:#f5f7fb; padding:24px 0; }
  .container { max-width:760px; margin:0 auto; background:#ffffff; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,0.06); }
  .header { background:#0a4e74; color:#ffffff; padding:20px 24px; }
  .header h1 { margin:0; font-size:22px; line-height:1.3; }
  .header p { margin:6px 0 0; font-size:13px; opacity:0.9; }
  .content { padding:24px; color:#2a2f36; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  h2 { font-size:18px; margin:26px 0 10px; color:#0a4e74; }
  h3 { font-size:16px; margin:18px 0 8px; color:#1f3b4d; }
  h4 { font-size:15px; margin:14px 0 6px; color:#2a4a5c; font-weight:600; }
  p { font-size:14px; line-height:1.6; margin:0 0 12px; }
  ul { margin:8px 0 16px 18px; padding:0; }
  li { margin:6px 0; font-size:14px; }
  .pill { display:inline-block; font-size:12px; padding:4px 10px; border-radius:999px; background:#eaf3f8; color:#0a4e74; vertical-align:middle; }
  .note { font-size:12px; color:#5b6570; margin-top:6px; }
  a { color:#0a4e74; text-decoration:none; }
  a:hover { text-decoration:underline; }
  .sep { height:1px; background:#eef0f4; margin:20px 0; }
  .company { border-left:4px solid #eaf3f8; padding-left:14px; margin-bottom:18px; }
  .subsection { margin-bottom:24px; }
  .subsection-header { background:#f8f9fa; padding:8px 12px; margin:16px -12px 12px -12px; border-left:4px solid #0a4e74; }
  .empty { color:#6b7480; font-style:italic; }
  .footer { padding:18px 24px 28px; font-size:12px; color:#6b7480; }
  .score-high { color:#28a745; font-weight:bold; }
  .score-medium { color:#ffc107; font-weight:bold; }
  .score-low { color:#dc3545; font-weight:bold; }
</style>'''

    def _generate_industry_section(self, industry_articles: List[Article]) -> List[str]:
        """Generate the Industry Update section with subsections"""
        html_parts = ['<h2>Industry Update</h2>']
        
        if not industry_articles:
            html_parts.append('<p class="empty">No recent industry news within the specified window.</p>')
            return html_parts
        
        # Categorize articles
        obesity_glp1_articles = []
        gene_cell_articles = []
        general_articles = []
        
        for article in industry_articles:
            category = self._categorize_article(article)
            if category == "Obesity/GLP-1":
                obesity_glp1_articles.append(article)
            elif category == "Cell/Gene Therapy":
                gene_cell_articles.append(article)
            else:
                general_articles.append(article)
        
        # Sort by relevance score
        obesity_glp1_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        gene_cell_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        general_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Generate subsections
        sections = [
            ("Obesity/GLP-1", obesity_glp1_articles),
            ("Cell/Gene Therapy", gene_cell_articles),
            ("General", general_articles)
        ]
        
        for section_name, articles in sections:
            html_parts.extend([
                '<div class="subsection">',
                f'<div class="subsection-header"><h4>{section_name}</h4></div>'
            ])
            
            if articles:
                for article in articles[:3]:  # Show top 3
                    html_parts.extend([
                        f'<h3><a href="{article.link}" target="_blank" rel="noopener">{article.title}</a></h3>',
                        f'<p>{article.summary}</p>'
                    ])
                    if article.relevance_score > 2.0:
                        score_class = self._get_score_class(article.relevance_score)
                        html_parts.append(f'<p class="note">Source: {article.source_name} • Relevance: <span class="{score_class}">{article.relevance_score:.1f}</span></p>')
            else:
                html_parts.append(f'<p class="empty">No recent {section_name} news.</p>')
            
            html_parts.append('</div>')
        
        return html_parts

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value"""
        if score >= 5.0:
            return 'score-high'
        elif score >= 2.0:
            return 'score-medium'
        else:
            return 'score-low'

    def _generate_company_section(self, updates: Dict[str, List[Article]]) -> List[str]:
        """Generate the Company Updates section"""
        html_parts = []
        
        company_mapping = {
            'ABOS': 'Acumen Pharmaceuticals (ABOS)',
            'ALXO': 'ALX Oncology (ALXO)', 
            'AMGN': 'Amgen (AMGN)',
            'REGN': 'Regeneron (REGN)',
            'JSPR': 'Jasper Therapeutics (JSPR)'
        }
        
        all_companies = set(company_mapping.keys())
        coverage_pills = ' • '.join(sorted(all_companies))
        
        html_parts.append(f'<h2>Company Updates <span class="pill">Coverage: {coverage_pills}</span></h2>')
        
        for ticker in sorted(company_mapping.keys()):
            company_name = company_mapping[ticker]
            articles = updates.get(ticker, [])
            
            html_parts.append('<div class="company">')
            
            if articles:
                sorted_articles = sorted(articles, key=lambda x: (x.relevance_score, x.published), reverse=True)
                main_article = sorted_articles[0]
                
                html_parts.extend([
                    f'<h3><a href="{main_article.link}" target="_blank" rel="noopener">{company_name} — {main_article.title}</a></h3>',
                    f'<p>{main_article.summary}</p>'
                ])
                
                if len(sorted_articles) > 1:
                    html_parts.append('<ul>')
                    for article in sorted_articles[1:4]:
                        html_parts.append(f'<li><a href="{article.link}" target="_blank" rel="noopener">{article.title}</a></li>')
                    html_parts.append('</ul>')
                
                if main_article.relevance_score > 3.0:
                    score_class = self._get_score_class(main_article.relevance_score)
                    html_parts.append(f'<p class="note">Relevance: <span class="{score_class}">{main_article.relevance_score:.1f}</span> • Published: {main_article.published.strftime("%b %d")}</p>')
            else:
                html_parts.extend([
                    f'<h3>{company_name}</h3>',
                    '<p class="empty">No new company disclosures or material news identified in the last 3 days.</p>'
                ])
            
            html_parts.append('</div>')
        
        return html_parts

    def send_email_report(self, html_report: str) -> bool:
        """Send email report"""
        email_config = self.config.get('email', {})
        
        if not email_config.get('enabled', True):
            logging.info("Email sending disabled in configuration")
            return False
        
        required_fields = ['smtp_host', 'smtp_port', 'sender', 'recipients']
        for field in required_fields:
            if not email_config.get(field):
                logging.error(f"Missing email configuration: {field}")
                return False
        
        try:
            msg = EmailMessage()
            msg['Subject'] = email_config['subject'].format(date=datetime.now(self.tz).strftime('%Y-%m-%d'))
            msg['From'] = email_config['sender']
            msg['To'] = email_config['recipients']
            
            msg.set_content("This email contains an HTML biotech update report.")
            msg.add_alternative(html_report, subtype='html')
            
            smtp_class = smtplib.SMTP_SSL if email_config['smtp_port'] == 465 else smtplib.SMTP
            
            with smtp_class(email_config['smtp_host'], email_config['smtp_port']) as server:
                if email_config['smtp_port'] != 465:
                    server.starttls()
                
                if email_config.get('smtp_user') and email_config.get('smtp_password'):
                    server.login(email_config['smtp_user'], email_config['smtp_password'])
                
                server.send_message(msg)
            
            logging.info(f"Email sent successfully to {len(email_config['recipients'])} recipients")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email: {e}")
            return False

    def run_full_workflow(self) -> bool:
        """Run the complete workflow: fetch, process, generate, and send"""
        try:
            logging.info("Starting full biotech update workflow")
            
            # Fetch updates from all sources
            updates = self.fetch_all_updates()
            
            total_articles = sum(len(articles) for articles in updates.values())
            
            if not any(updates.values()):
                logging.warning("No new articles found across all sources")
                return False
            
            # Generate HTML report
            html_report = self.generate_html_report(updates)
            
            # Send email report
            email_sent = self.send_email_report(html_report)
            
            logging.info(f"Workflow completed: {total_articles} articles processed, email sent: {email_sent}")
            return True
            
        except Exception as e:
            logging.error(f"Workflow failed: {e}")
            return False

# ----------------------------------------------------------------------
# Utility Functions

def create_simple_env_template():
    """Create a simple .env template for manual editing"""
    env_content = """# ===============================================================
# BIOTECH DAILY DIGEST - ENVIRONMENT CONFIGURATION
# ===============================================================

# EMAIL SETTINGS (Required for sending reports)
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.email@gmail.com
SMTP_PASSWORD=your-gmail-app-password
REPORT_SENDER=your.email@gmail.com
REPORT_RECIPIENTS=recipient1@example.com,recipient2@example.com
REPORT_SUBJECT=Daily Biotech Update - {date}

# WEBHOOK NOTIFICATIONS (Optional)
WEBHOOK_ENABLED=false
WEBHOOK_URL=
WEBHOOK_TYPE=slack

# CONTENT FILTERING
MIN_WORD_COUNT=10
EXCLUDED_KEYWORDS=spam,advertisement,promotional
RELEVANCE_THRESHOLD=2.0

# FEATURE SETTINGS
ENABLE_DASHBOARD=true
ENABLE_ANALYTICS=true

# FILE PATHS
SOURCES_FILE=data_sources.json
"""
    return env_content

def create_sample_sources():
    """Create a sample data sources file"""
    sample_sources = [
        {
            "name": "Novo Nordisk News",
            "category": "Industry",
            "type": "rss",
            "url": "https://www.novonordisk.com/bin/rss/news.xml",
            "enabled": True,
            "keywords": ["novo nordisk", "ozempic", "wegovy", "semaglutide", "glp-1", "obesity", "diabetes"]
        },
        {
            "name": "Eli Lilly News",
            "category": "Industry",
            "type": "rss",
            "url": "https://investor.lilly.com/rss/news-releases",
            "enabled": True,
            "keywords": ["eli lilly", "mounjaro", "zepbound", "tirzepatide", "glp-1", "gip", "obesity", "diabetes"]
        },
        {
            "name": "BioPharma Dive",
            "category": "Industry",
            "type": "rss",
            "url": "https://www.biopharmadive.com/feeds/news/",
            "enabled": True,
            "keywords": ["biotech", "pharmaceutical", "fda", "glp-1", "obesity"]
        },
        {
            "name": "FierceBiotech",
            "category": "Industry",
            "type": "rss",
            "url": "https://www.fiercebiotech.com/rss/xml",
            "enabled": True,
            "keywords": ["biotech", "clinical trial", "obesity", "metabolic"]
        }
    ]
    
    with open('data_sources.json', 'w') as f:
        json.dump(sample_sources, f, indent=2)
    
    print("Sample data sources created as 'data_sources.json'")

def main():
    """Main entry point with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Daily Biotech Update Script')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--create-sources', action='store_true', help='Create sample data sources file')
    parser.add_argument('--create-env-template', action='store_true', help='Create .env template')
    parser.add_argument('--test-email', action='store_true', help='Send test email')
    
    args = parser.parse_args()
    
    if args.create_sources:
        create_sample_sources()
        return
        
    if args.create_env_template:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(create_simple_env_template())
        print("Created .env template")
        return
    
    # Initialize manager
    try:
        manager = BiotechUpdateManager(args.config)
    except Exception as e:
        print(f"Failed to initialize manager: {e}")
        return
    
    if args.test_email:
        test_html = """
        <html><body>
        <h1>Test Email</h1>
        <p>This is a test email from the Biotech Update System.</p>
        <p>If you received this, your email configuration is working correctly.</p>
        </body></html>
        """
        success = manager.send_email_report(test_html)
        print(f"Test email {'sent successfully' if success else 'failed to send'}")
        return
    
    # Run full workflow
    success = manager.run_full_workflow()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()