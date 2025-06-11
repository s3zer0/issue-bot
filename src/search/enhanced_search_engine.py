"""
Enhanced Issue Search Engine

Comprehensive multi-source search system with advanced capabilities for discovering
50-100 relevant issues per search with rich metadata and intelligent analysis.
"""

import asyncio
import aiohttp
import time
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from urllib.parse import quote, urlparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from loguru import logger

# For NLP processing
try:
    import openai
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from textblob import TextBlob
except ImportError as e:
    logger.warning(f"Optional dependencies not available: {e}")

from src.models import IssueItem, SearchResult
from src.config import config


@dataclass
class SourceInfo:
    """Information about a search source."""
    name: str
    type: str  # news, academic, social, technical, financial
    tier: int  # 1=official, 2=verified, 3=community
    base_url: str
    api_endpoint: Optional[str] = None
    rate_limit: int = 100  # requests per hour
    reliability_score: float = 0.5
    language: str = "en"
    requires_auth: bool = False


@dataclass
class EnhancedIssue:
    """Enhanced issue with comprehensive metadata."""
    # Core content
    title: str
    summary: str
    content: str
    url: str
    
    # Source information
    source: SourceInfo
    published_date: datetime
    author: Optional[str] = None
    
    # Scoring and analysis
    relevance_score: float = 0.0
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_confidence: float = 0.0
    impact_score: float = 0.0
    controversy_score: float = 0.0
    momentum_score: float = 0.0  # trend momentum
    authority_score: float = 0.0  # source authority
    
    # Social signals
    shares: int = 0
    comments: int = 0
    likes: int = 0
    views: int = 0
    
    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # named entities
    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    
    # Validation
    fact_check_score: float = 0.0
    cross_reference_count: int = 0
    contradiction_flags: List[str] = field(default_factory=list)
    
    # Processing metadata
    discovery_method: str = "unknown"
    processing_time: float = 0.0
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


class SourceManager(ABC):
    """Abstract base class for source managers."""
    
    def __init__(self, source_info: SourceInfo):
        self.source_info = source_info
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[EnhancedIssue]:
        """Search for issues using this source."""
        pass
    
    async def rate_limit_wait(self):
        """Implement rate limiting."""
        if self.request_count >= self.source_info.rate_limit:
            wait_time = 3600 - (time.time() - self.last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = time.time()


class NewsSourceManager(SourceManager):
    """Manager for news sources."""
    
    def __init__(self):
        source_info = SourceInfo(
            name="News Aggregator",
            type="news",
            tier=2,
            base_url="https://newsapi.org",
            api_endpoint="/v2/everything",
            rate_limit=1000,
            reliability_score=0.8
        )
        super().__init__(source_info)
    
    async def search(self, query: str, limit: int = 10) -> List[EnhancedIssue]:
        """Search news sources."""
        await self.rate_limit_wait()
        
        issues = []
        try:
            # Simulate news API call
            # In production, replace with actual news API calls
            news_results = await self._fetch_news_api(query, limit)
            
            for article in news_results:
                issue = await self._convert_article_to_issue(article)
                if issue:
                    issues.append(issue)
                    
        except Exception as e:
            logger.error(f"News search failed for '{query}': {e}")
        
        return issues
    
    async def _fetch_news_api(self, query: str, limit: int) -> List[Dict]:
        """Fetch from news API."""
        # Mock implementation - replace with actual API calls
        return [
            {
                "title": f"Latest developments in {query}",
                "description": f"Recent news about {query} and its implications",
                "url": f"https://example-news.com/article-{i}",
                "publishedAt": datetime.now().isoformat(),
                "source": {"name": "TechNews"},
                "content": f"Detailed content about {query}..."
            }
            for i in range(min(limit, 5))
        ]
    
    async def _convert_article_to_issue(self, article: Dict) -> Optional[EnhancedIssue]:
        """Convert news article to enhanced issue."""
        try:
            return EnhancedIssue(
                title=article["title"],
                summary=article["description"],
                content=article.get("content", article["description"]),
                url=article["url"],
                source=self.source_info,
                published_date=datetime.fromisoformat(article["publishedAt"].replace('Z', '+00:00')),
                author=article.get("author"),
                category="news",
                discovery_method="news_api"
            )
        except Exception as e:
            logger.error(f"Failed to convert article: {e}")
            return None


class AcademicSourceManager(SourceManager):
    """Manager for academic sources."""
    
    def __init__(self):
        source_info = SourceInfo(
            name="Academic Papers",
            type="academic",
            tier=1,
            base_url="https://api.semanticscholar.org",
            rate_limit=100,
            reliability_score=0.9
        )
        super().__init__(source_info)
    
    async def search(self, query: str, limit: int = 10) -> List[EnhancedIssue]:
        """Search academic sources."""
        await self.rate_limit_wait()
        
        issues = []
        try:
            # Search multiple academic databases
            sources = [
                self._search_semantic_scholar(query, limit//3),
                self._search_arxiv(query, limit//3),
                self._search_pubmed(query, limit//3)
            ]
            
            results = await asyncio.gather(*sources, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    issues.extend(result)
                    
        except Exception as e:
            logger.error(f"Academic search failed for '{query}': {e}")
        
        return issues
    
    async def _search_semantic_scholar(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search Semantic Scholar."""
        # Mock implementation
        return [
            EnhancedIssue(
                title=f"Research on {query}: A Comprehensive Study",
                summary=f"Academic research investigating {query} with novel findings",
                content=f"Abstract: This paper presents new research on {query}...",
                url=f"https://semanticscholar.org/paper/{i}",
                source=self.source_info,
                published_date=datetime.now() - timedelta(days=30),
                category="academic",
                authority_score=0.9,
                discovery_method="semantic_scholar"
            )
            for i in range(min(limit, 3))
        ]
    
    async def _search_arxiv(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search arXiv preprints."""
        # Mock implementation
        return []
    
    async def _search_pubmed(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search PubMed for medical research."""
        # Mock implementation
        return []


class SocialMediaManager(SourceManager):
    """Manager for social media sources."""
    
    def __init__(self):
        source_info = SourceInfo(
            name="Social Media",
            type="social",
            tier=3,
            base_url="https://api.twitter.com",
            rate_limit=300,
            reliability_score=0.4
        )
        super().__init__(source_info)
    
    async def search(self, query: str, limit: int = 10) -> List[EnhancedIssue]:
        """Search social media sources."""
        await self.rate_limit_wait()
        
        issues = []
        try:
            # Search multiple social platforms
            sources = [
                self._search_twitter(query, limit//4),
                self._search_reddit(query, limit//4),
                self._search_linkedin(query, limit//4),
                self._search_youtube(query, limit//4)
            ]
            
            results = await asyncio.gather(*sources, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    issues.extend(result)
                    
        except Exception as e:
            logger.error(f"Social media search failed for '{query}': {e}")
        
        return issues
    
    async def _search_twitter(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search Twitter/X."""
        # Mock implementation
        return [
            EnhancedIssue(
                title=f"Twitter discussion on {query}",
                summary=f"Social media buzz around {query}",
                content=f"Multiple tweets discussing {query} trends...",
                url=f"https://twitter.com/search?q={quote(query)}",
                source=self.source_info,
                published_date=datetime.now() - timedelta(hours=2),
                category="social",
                shares=150,
                comments=45,
                likes=320,
                discovery_method="twitter_api"
            )
            for i in range(min(limit, 2))
        ]
    
    async def _search_reddit(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search Reddit discussions."""
        # Mock implementation
        return []
    
    async def _search_linkedin(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search LinkedIn posts."""
        # Mock implementation
        return []
    
    async def _search_youtube(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search YouTube content."""
        # Mock implementation
        return []


class TechnicalSourceManager(SourceManager):
    """Manager for technical sources."""
    
    def __init__(self):
        source_info = SourceInfo(
            name="Technical Sources",
            type="technical",
            tier=2,
            base_url="https://api.github.com",
            rate_limit=5000,
            reliability_score=0.7
        )
        super().__init__(source_info)
    
    async def search(self, query: str, limit: int = 10) -> List[EnhancedIssue]:
        """Search technical sources."""
        await self.rate_limit_wait()
        
        issues = []
        try:
            # Search multiple technical platforms
            sources = [
                self._search_github(query, limit//4),
                self._search_stackoverflow(query, limit//4),
                self._search_hacker_news(query, limit//4),
                self._search_tech_blogs(query, limit//4)
            ]
            
            results = await asyncio.gather(*sources, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    issues.extend(result)
                    
        except Exception as e:
            logger.error(f"Technical search failed for '{query}': {e}")
        
        return issues
    
    async def _search_github(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search GitHub repositories and issues."""
        # Mock implementation
        return [
            EnhancedIssue(
                title=f"GitHub project: {query} implementation",
                summary=f"Open source implementation and discussions about {query}",
                content=f"Repository containing {query} code and documentation...",
                url=f"https://github.com/search?q={quote(query)}",
                source=self.source_info,
                published_date=datetime.now() - timedelta(days=7),
                category="technical",
                authority_score=0.8,
                discovery_method="github_api"
            )
            for i in range(min(limit, 2))
        ]
    
    async def _search_stackoverflow(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search Stack Overflow."""
        # Mock implementation
        return []
    
    async def _search_hacker_news(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search Hacker News."""
        # Mock implementation
        return []
    
    async def _search_tech_blogs(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search technical blogs."""
        # Mock implementation
        return []


class FinancialSourceManager(SourceManager):
    """Manager for financial sources."""
    
    def __init__(self):
        source_info = SourceInfo(
            name="Financial Sources",
            type="financial",
            tier=1,
            base_url="https://api.sec.gov",
            rate_limit=10,
            reliability_score=0.95
        )
        super().__init__(source_info)
    
    async def search(self, query: str, limit: int = 10) -> List[EnhancedIssue]:
        """Search financial sources."""
        await self.rate_limit_wait()
        
        issues = []
        try:
            # Search financial databases
            sources = [
                self._search_sec_filings(query, limit//3),
                self._search_earnings_calls(query, limit//3),
                self._search_financial_news(query, limit//3)
            ]
            
            results = await asyncio.gather(*sources, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    issues.extend(result)
                    
        except Exception as e:
            logger.error(f"Financial search failed for '{query}': {e}")
        
        return issues
    
    async def _search_sec_filings(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search SEC filings."""
        # Mock implementation
        return []
    
    async def _search_earnings_calls(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search earnings call transcripts."""
        # Mock implementation
        return []
    
    async def _search_financial_news(self, query: str, limit: int) -> List[EnhancedIssue]:
        """Search financial news."""
        # Mock implementation
        return []


class AdvancedQueryGenerator:
    """Advanced query generation engine."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
        self.synonyms_cache = {}
        self.embeddings_model = None
        
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load embeddings model: {e}")
    
    async def generate_query_variations(self, topic: str, source_type: str = "general") -> List[str]:
        """Generate 20-30 search variations for a topic."""
        queries = []
        
        # Base query
        queries.append(topic)
        
        # Synonym expansion
        queries.extend(await self._generate_synonym_variations(topic))
        
        # Technical vs layman
        queries.extend(await self._generate_technical_variations(topic, source_type))
        
        # Temporal variations
        queries.extend(self._generate_temporal_variations(topic))
        
        # Geographic variations
        queries.extend(self._generate_geographic_variations(topic))
        
        # Acronym variations
        queries.extend(self._generate_acronym_variations(topic))
        
        # Common misspellings
        queries.extend(self._generate_misspelling_variations(topic))
        
        # Hypothetical scenarios
        queries.extend(await self._generate_scenario_variations(topic))
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))
        return unique_queries[:30]
    
    async def _generate_synonym_variations(self, topic: str) -> List[str]:
        """Generate synonym-based variations."""
        if topic in self.synonyms_cache:
            return self.synonyms_cache[topic]
        
        variations = []
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 8-10 synonym variations and related terms for search queries. Return only the variations, one per line."
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}"
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            variations = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
            self.synonyms_cache[topic] = variations
            
        except Exception as e:
            logger.error(f"Synonym generation failed: {e}")
            # Fallback to simple variations
            variations = [
                f"{topic} technology",
                f"{topic} innovation",
                f"{topic} development",
                f"{topic} advancement"
            ]
        
        return variations
    
    async def _generate_technical_variations(self, topic: str, source_type: str) -> List[str]:
        """Generate technical vs layman variations."""
        variations = []
        
        if source_type == "technical":
            variations.extend([
                f"{topic} implementation",
                f"{topic} architecture",
                f"{topic} algorithm",
                f"{topic} protocol",
                f"{topic} API",
                f"{topic} framework"
            ])
        elif source_type == "social":
            variations.extend([
                f"What is {topic}?",
                f"How does {topic} work?",
                f"{topic} explained",
                f"{topic} for beginners",
                f"{topic} impact",
                f"{topic} controversy"
            ])
        elif source_type == "academic":
            variations.extend([
                f"{topic} research",
                f"{topic} study",
                f"{topic} analysis",
                f"{topic} methodology",
                f"{topic} theory",
                f"{topic} empirical"
            ])
        
        return variations
    
    def _generate_temporal_variations(self, topic: str) -> List[str]:
        """Generate time-based variations."""
        return [
            f"latest {topic}",
            f"recent {topic}",
            f"{topic} 2024",
            f"upcoming {topic}",
            f"future of {topic}",
            f"{topic} trends",
            f"new {topic}",
            f"{topic} update"
        ]
    
    def _generate_geographic_variations(self, topic: str) -> List[str]:
        """Generate geographic variations."""
        regions = ["global", "US", "Europe", "Asia", "China", "Korea", "Japan"]
        return [f"{topic} {region}" for region in regions]
    
    def _generate_acronym_variations(self, topic: str) -> List[str]:
        """Generate acronym and full form variations."""
        # Common tech acronyms
        acronym_map = {
            "artificial intelligence": ["AI", "machine learning", "ML"],
            "machine learning": ["ML", "AI", "artificial intelligence"],
            "virtual reality": ["VR", "augmented reality", "AR"],
            "internet of things": ["IoT", "connected devices"],
            "application programming interface": ["API", "interface"],
            "software as a service": ["SaaS", "cloud software"],
        }
        
        topic_lower = topic.lower()
        variations = []
        
        for key, values in acronym_map.items():
            if key in topic_lower:
                variations.extend(values)
            elif any(v.lower() in topic_lower for v in values):
                variations.extend([key] + [v for v in values if v.lower() not in topic_lower])
        
        return variations
    
    def _generate_misspelling_variations(self, topic: str) -> List[str]:
        """Generate common misspelling variations."""
        # Simple character substitutions for common errors
        variations = []
        words = topic.split()
        
        for word in words:
            if len(word) > 4:
                # Common substitutions
                variations.append(word.replace('ei', 'ie'))
                variations.append(word.replace('ie', 'ei'))
                variations.append(word.replace('ph', 'f'))
                variations.append(word.replace('f', 'ph'))
        
        return [' '.join(words).replace(old, new) for old, new in zip(topic.split(), variations) if old != new]
    
    async def _generate_scenario_variations(self, topic: str) -> List[str]:
        """Generate hypothetical future scenarios."""
        scenarios = [
            f"what if {topic}",
            f"{topic} impact on society",
            f"{topic} risks and benefits",
            f"{topic} adoption challenges",
            f"{topic} market disruption",
            f"{topic} regulation",
            f"{topic} ethical concerns"
        ]
        return scenarios


class SemanticDeduplicator:
    """Intelligent deduplication using semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embeddings_model = None
        
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load embeddings model: {e}")
    
    async def deduplicate_and_cluster(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Deduplicate issues and assign cluster IDs."""
        if not self.embeddings_model or len(issues) < 2:
            return issues
        
        try:
            # Generate embeddings
            texts = [f"{issue.title} {issue.summary}" for issue in issues]
            embeddings = self.embeddings_model.encode(texts)
            
            # Store embeddings in issues
            for issue, embedding in zip(issues, embeddings):
                issue.embedding = embedding
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,
                min_samples=1,
                metric='precomputed'
            ).fit(1 - similarity_matrix)
            
            # Assign cluster IDs
            for issue, cluster_id in zip(issues, clustering.labels_):
                issue.cluster_id = cluster_id
            
            # Select best representative from each cluster
            deduplicated = self._select_cluster_representatives(issues, clustering.labels_)
            
            logger.info(f"Deduplicated {len(issues)} issues to {len(deduplicated)}")
            return deduplicated
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return issues
    
    def _select_cluster_representatives(self, issues: List[EnhancedIssue], cluster_labels: np.ndarray) -> List[EnhancedIssue]:
        """Select the best representative from each cluster."""
        clusters = {}
        
        # Group issues by cluster
        for issue, cluster_id in zip(issues, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(issue)
        
        representatives = []
        
        for cluster_id, cluster_issues in clusters.items():
            if len(cluster_issues) == 1:
                representatives.append(cluster_issues[0])
            else:
                # Select best representative based on multiple factors
                best_issue = max(cluster_issues, key=self._calculate_representative_score)
                
                # Merge information from other issues in cluster
                self._merge_cluster_information(best_issue, cluster_issues)
                representatives.append(best_issue)
        
        return representatives
    
    def _calculate_representative_score(self, issue: EnhancedIssue) -> float:
        """Calculate score for selecting cluster representative."""
        score = 0.0
        
        # Source reliability
        score += issue.source.reliability_score * 0.3
        
        # Authority score
        score += issue.authority_score * 0.2
        
        # Social signals
        social_score = (issue.shares + issue.comments + issue.likes) / 1000
        score += min(social_score, 1.0) * 0.2
        
        # Content quality (length and detail)
        content_score = min(len(issue.content) / 1000, 1.0)
        score += content_score * 0.2
        
        # Recency
        days_old = (datetime.now() - issue.published_date).days
        recency_score = max(0, 1 - days_old / 30)  # Decay over 30 days
        score += recency_score * 0.1
        
        return score
    
    def _merge_cluster_information(self, representative: EnhancedIssue, cluster_issues: List[EnhancedIssue]):
        """Merge information from cluster into representative."""
        # Combine cross-reference count
        representative.cross_reference_count = len(cluster_issues) - 1
        
        # Merge tags and keywords
        all_tags = set(representative.tags)
        all_keywords = set(representative.keywords)
        
        for issue in cluster_issues:
            if issue != representative:
                all_tags.update(issue.tags)
                all_keywords.update(issue.keywords)
        
        representative.tags = list(all_tags)
        representative.keywords = list(all_keywords)
        
        # Average sentiment scores
        sentiment_scores = [issue.sentiment_score for issue in cluster_issues if issue.sentiment_score != 0]
        if sentiment_scores:
            representative.sentiment_score = np.mean(sentiment_scores)
        
        # Sum social signals
        representative.shares = sum(issue.shares for issue in cluster_issues)
        representative.comments = sum(issue.comments for issue in cluster_issues)
        representative.likes = sum(issue.likes for issue in cluster_issues)


class EnhancedRelevanceScorer:
    """Multi-factor relevance scoring system."""
    
    def __init__(self):
        self.weights = {
            'semantic_similarity': 0.25,
            'temporal_relevance': 0.15,
            'source_authority': 0.15,
            'social_signals': 0.10,
            'geographic_relevance': 0.10,
            'impact_assessment': 0.15,
            'keyword_match': 0.10
        }
    
    async def score_issues(self, issues: List[EnhancedIssue], original_topic: str) -> List[EnhancedIssue]:
        """Score all issues for relevance."""
        try:
            for issue in issues:
                issue.relevance_score = await self._calculate_relevance_score(issue, original_topic)
            
            # Sort by relevance score
            issues.sort(key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}")
        
        return issues
    
    async def _calculate_relevance_score(self, issue: EnhancedIssue, original_topic: str) -> float:
        """Calculate comprehensive relevance score."""
        score = 0.0
        
        # Semantic similarity
        semantic_score = await self._calculate_semantic_similarity(issue, original_topic)
        score += semantic_score * self.weights['semantic_similarity']
        
        # Temporal relevance
        temporal_score = self._calculate_temporal_relevance(issue)
        score += temporal_score * self.weights['temporal_relevance']
        
        # Source authority
        source_score = issue.source.reliability_score
        score += source_score * self.weights['source_authority']
        
        # Social signals
        social_score = self._calculate_social_score(issue)
        score += social_score * self.weights['social_signals']
        
        # Geographic relevance (simplified)
        geo_score = 0.5  # Default neutral
        score += geo_score * self.weights['geographic_relevance']
        
        # Impact assessment
        impact_score = issue.impact_score
        score += impact_score * self.weights['impact_assessment']
        
        # Keyword match
        keyword_score = self._calculate_keyword_match(issue, original_topic)
        score += keyword_score * self.weights['keyword_match']
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _calculate_semantic_similarity(self, issue: EnhancedIssue, original_topic: str) -> float:
        """Calculate semantic similarity between issue and original topic."""
        try:
            if hasattr(issue, 'embedding') and issue.embedding is not None:
                # Use stored embedding if available
                model = SentenceTransformer('all-MiniLM-L6-v2')
                topic_embedding = model.encode([original_topic])
                similarity = cosine_similarity([issue.embedding], topic_embedding)[0][0]
                return max(0, similarity)
            else:
                # Fallback to simple text matching
                return self._simple_text_similarity(issue, original_topic)
        except Exception:
            return self._simple_text_similarity(issue, original_topic)
    
    def _simple_text_similarity(self, issue: EnhancedIssue, original_topic: str) -> float:
        """Simple text similarity fallback."""
        text = f"{issue.title} {issue.summary}".lower()
        topic_words = set(original_topic.lower().split())
        text_words = set(text.split())
        
        if not topic_words:
            return 0.0
        
        intersection = topic_words.intersection(text_words)
        return len(intersection) / len(topic_words)
    
    def _calculate_temporal_relevance(self, issue: EnhancedIssue) -> float:
        """Calculate temporal relevance with exponential decay."""
        days_old = (datetime.now() - issue.published_date).days
        
        # Exponential decay with half-life of 7 days
        half_life = 7
        decay_factor = 0.5 ** (days_old / half_life)
        
        return min(decay_factor, 1.0)
    
    def _calculate_social_score(self, issue: EnhancedIssue) -> float:
        """Calculate social signals score."""
        total_engagement = issue.shares + issue.comments + issue.likes
        
        if total_engagement == 0:
            return 0.1  # Neutral score for no engagement
        
        # Logarithmic scaling for engagement
        import math
        score = math.log10(total_engagement + 1) / 4  # Scale to roughly 0-1
        return min(score, 1.0)
    
    def _calculate_keyword_match(self, issue: EnhancedIssue, original_topic: str) -> float:
        """Calculate keyword matching score."""
        topic_keywords = set(original_topic.lower().split())
        issue_keywords = set(issue.keywords + issue.title.lower().split())
        
        if not topic_keywords:
            return 0.0
        
        matches = topic_keywords.intersection(issue_keywords)
        return len(matches) / len(topic_keywords)


class SentimentAnalyzer:
    """Sentiment and impact analysis system."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
    
    async def analyze_sentiment_and_impact(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Analyze sentiment and impact for all issues."""
        try:
            # Process in batches to manage API costs
            batch_size = 10
            for i in range(0, len(issues), batch_size):
                batch = issues[i:i + batch_size]
                await self._process_sentiment_batch(batch)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
        
        return issues
    
    async def _process_sentiment_batch(self, issues: List[EnhancedIssue]):
        """Process a batch of issues for sentiment analysis."""
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            for issue in issues:
                # Simple sentiment analysis
                try:
                    blob = TextBlob(issue.content)
                    issue.sentiment_score = blob.sentiment.polarity
                    issue.sentiment_confidence = abs(blob.sentiment.polarity)
                except Exception:
                    issue.sentiment_score = 0.0
                    issue.sentiment_confidence = 0.0
                
                # Calculate impact score based on multiple factors
                issue.impact_score = self._calculate_impact_score(issue)
                
                # Calculate controversy score
                issue.controversy_score = self._calculate_controversy_score(issue)
                
                # Calculate momentum score
                issue.momentum_score = self._calculate_momentum_score(issue)
                
        except Exception as e:
            logger.error(f"Batch sentiment processing failed: {e}")
    
    def _calculate_impact_score(self, issue: EnhancedIssue) -> float:
        """Calculate potential impact score."""
        score = 0.0
        
        # Source tier importance
        if issue.source.tier == 1:
            score += 0.4
        elif issue.source.tier == 2:
            score += 0.3
        else:
            score += 0.1
        
        # Social engagement
        engagement = issue.shares + issue.comments + issue.likes
        engagement_score = min(engagement / 1000, 0.3)
        score += engagement_score
        
        # Content indicators
        impact_keywords = ['breaking', 'major', 'significant', 'important', 'critical', 'urgent']
        content_lower = issue.content.lower()
        keyword_matches = sum(1 for keyword in impact_keywords if keyword in content_lower)
        score += min(keyword_matches * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_controversy_score(self, issue: EnhancedIssue) -> float:
        """Calculate controversy level."""
        # For now, use simple heuristics
        # In production, this could analyze comment sentiment variance
        
        controversy_keywords = ['debate', 'controversy', 'disputed', 'conflict', 'criticism']
        content_lower = issue.content.lower()
        
        matches = sum(1 for keyword in controversy_keywords if keyword in content_lower)
        return min(matches * 0.2, 1.0)
    
    def _calculate_momentum_score(self, issue: EnhancedIssue) -> float:
        """Calculate trend momentum."""
        # Simple implementation based on recency and engagement
        days_old = (datetime.now() - issue.published_date).days
        
        if days_old <= 1:
            momentum = 1.0
        elif days_old <= 7:
            momentum = 0.8
        elif days_old <= 30:
            momentum = 0.5
        else:
            momentum = 0.2
        
        # Boost for high engagement
        engagement_boost = min((issue.shares + issue.comments) / 500, 0.3)
        
        return min(momentum + engagement_boost, 1.0)