"""
Enhanced Issue Searcher - Main Orchestrator

Comprehensive issue discovery system that coordinates all search components
to find 50-100 relevant issues with rich metadata and high confidence scores.
"""

import asyncio
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from loguru import logger

from src.search.enhanced_search_engine import (
    EnhancedIssue,
    SourceManager,
    NewsSourceManager,
    AcademicSourceManager,
    SocialMediaManager,
    TechnicalSourceManager,
    FinancialSourceManager,
    AdvancedQueryGenerator,
    SemanticDeduplicator,
    EnhancedRelevanceScorer,
    SentimentAnalyzer
)
from src.models import IssueItem, SearchResult
from src.config import config


class DeepWebMiner:
    """Deep web mining capabilities for extracting additional content."""
    
    def __init__(self):
        self.session = None
        self.processed_urls = set()
    
    async def mine_additional_content(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Mine additional content from issue URLs and references."""
        enhanced_issues = []
        
        for issue in issues:
            try:
                enhanced = await self._deep_mine_issue(issue)
                enhanced_issues.append(enhanced)
            except Exception as e:
                logger.error(f"Deep mining failed for {issue.url}: {e}")
                enhanced_issues.append(issue)
        
        return enhanced_issues
    
    async def _deep_mine_issue(self, issue: EnhancedIssue) -> EnhancedIssue:
        """Perform deep mining on a single issue."""
        # Extract entities and keywords
        issue.entities = await self._extract_entities(issue.content)
        issue.keywords = await self._extract_keywords(issue.content)
        
        # Follow references if available
        references = await self._extract_references(issue.content)
        if references:
            additional_content = await self._follow_references(references[:3])  # Limit to 3
            if additional_content:
                issue.content += f"\n\nAdditional Context:\n{additional_content}"
        
        # Extract multimedia metadata
        multimedia_data = await self._extract_multimedia_metadata(issue.url)
        if multimedia_data:
            issue.tags.extend(multimedia_data.get('tags', []))
        
        return issue
    
    async def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        # Simple entity extraction - in production use spaCy or similar
        entities = []
        
        # Extract potential company names (capitalized words)
        import re
        company_pattern = r'\b[A-Z][a-z]+ ?(?:[A-Z][a-z]+)*\b'
        companies = re.findall(company_pattern, content)
        entities.extend(companies[:10])  # Limit to 10
        
        # Extract potential dates
        date_pattern = r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b'
        dates = re.findall(date_pattern, content)
        entities.extend(dates)
        
        # Extract numbers with units
        number_pattern = r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|percent|%|USD|dollars?)\b'
        numbers = re.findall(number_pattern, content, re.IGNORECASE)
        entities.extend(numbers)
        
        return list(set(entities))
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content."""
        # Simple keyword extraction
        words = content.lower().split()
        
        # Filter for meaningful words (length > 3, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        keywords = [word.strip('.,!?;:') for word in words 
                   if len(word) > 3 and word.lower() not in common_words and word.isalpha()]
        
        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]
    
    async def _extract_references(self, content: str) -> List[str]:
        """Extract reference URLs from content."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        return [url for url in urls if url not in self.processed_urls]
    
    async def _follow_references(self, urls: List[str]) -> str:
        """Follow reference URLs and extract additional content."""
        additional_content = ""
        
        for url in urls:
            try:
                self.processed_urls.add(url)
                # Mock implementation - in production, fetch and parse URLs
                additional_content += f"Referenced content from {url}: Additional insights...\n"
            except Exception as e:
                logger.error(f"Failed to follow reference {url}: {e}")
        
        return additional_content
    
    async def _extract_multimedia_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from multimedia content."""
        # Mock implementation for video/audio transcription
        if 'youtube.com' in url or 'youtu.be' in url:
            return {
                'type': 'video',
                'tags': ['video_content', 'multimedia'],
                'transcript_available': True
            }
        elif any(domain in url for domain in ['podcast', 'spotify', 'soundcloud']):
            return {
                'type': 'audio',
                'tags': ['podcast', 'audio_content'],
                'transcript_available': False
            }
        
        return {}


class PredictiveAnalyzer:
    """Predictive analysis layer for trend forecasting."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
    
    async def generate_predictions(self, issues: List[EnhancedIssue], topic: str) -> Dict[str, Any]:
        """Generate predictive analysis based on discovered issues."""
        predictions = {
            'trend_direction': await self._predict_trend_direction(issues),
            'escalation_probability': await self._predict_escalation(issues),
            'potential_outcomes': await self._predict_outcomes(issues, topic),
            'key_factors': await self._identify_key_factors(issues),
            'timeline_forecast': await self._forecast_timeline(issues),
            'confidence_level': self._calculate_prediction_confidence(issues)
        }
        
        return predictions
    
    async def _predict_trend_direction(self, issues: List[EnhancedIssue]) -> str:
        """Predict overall trend direction."""
        if not issues:
            return "neutral"
        
        # Analyze sentiment trends over time
        recent_issues = [issue for issue in issues 
                        if (datetime.now() - issue.published_date).days <= 7]
        older_issues = [issue for issue in issues 
                       if (datetime.now() - issue.published_date).days > 7]
        
        if recent_issues and older_issues:
            recent_sentiment = np.mean([issue.sentiment_score for issue in recent_issues])
            older_sentiment = np.mean([issue.sentiment_score for issue in older_issues])
            
            if recent_sentiment > older_sentiment + 0.1:
                return "improving"
            elif recent_sentiment < older_sentiment - 0.1:
                return "declining"
        
        return "stable"
    
    async def _predict_escalation(self, issues: List[EnhancedIssue]) -> float:
        """Predict probability of issue escalation."""
        if not issues:
            return 0.0
        
        escalation_factors = 0.0
        
        # High controversy score
        avg_controversy = np.mean([issue.controversy_score for issue in issues])
        escalation_factors += avg_controversy * 0.3
        
        # Increasing momentum
        avg_momentum = np.mean([issue.momentum_score for issue in issues])
        escalation_factors += avg_momentum * 0.3
        
        # Social engagement growth
        high_engagement_issues = len([issue for issue in issues 
                                    if (issue.shares + issue.comments + issue.likes) > 100])
        engagement_factor = min(high_engagement_issues / len(issues), 1.0)
        escalation_factors += engagement_factor * 0.2
        
        # Source tier diversity (higher tier sources = higher escalation risk)
        tier1_issues = len([issue for issue in issues if issue.source.tier == 1])
        if tier1_issues > 0:
            escalation_factors += 0.2
        
        return min(escalation_factors, 1.0)
    
    async def _predict_outcomes(self, issues: List[EnhancedIssue], topic: str) -> List[str]:
        """Predict potential outcomes."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Prepare context from issues
            issue_summaries = "\n".join([
                f"- {issue.title}: {issue.summary}"
                for issue in issues[:10]
            ])
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trend analyst. Based on the provided issues, predict 3-5 potential outcomes or scenarios. Be specific and realistic."
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nCurrent Issues:\n{issue_summaries}\n\nPredict potential outcomes:"
                    }
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            outcomes_text = response.choices[0].message.content
            outcomes = [line.strip().lstrip('123456789.-') 
                       for line in outcomes_text.split('\n') 
                       if line.strip()]
            
            return outcomes[:5]
            
        except Exception as e:
            logger.error(f"Outcome prediction failed: {e}")
            return [
                "Continued development and adoption",
                "Potential regulatory intervention",
                "Market consolidation",
                "Technology maturation",
                "Public sentiment shift"
            ]
    
    async def _identify_key_factors(self, issues: List[EnhancedIssue]) -> List[str]:
        """Identify key factors influencing the topic."""
        factors = []
        
        # Extract most common entities across issues
        all_entities = []
        for issue in issues:
            all_entities.extend(issue.entities)
        
        from collections import Counter
        entity_counts = Counter(all_entities)
        factors.extend([entity for entity, count in entity_counts.most_common(5)])
        
        # Add high-impact issues as factors
        high_impact_issues = [issue for issue in issues if issue.impact_score > 0.7]
        factors.extend([issue.title[:50] for issue in high_impact_issues[:3]])
        
        return factors[:8]
    
    async def _forecast_timeline(self, issues: List[EnhancedIssue]) -> Dict[str, str]:
        """Forecast timeline of developments."""
        timeline = {
            "immediate": "Current issues and immediate developments",
            "short_term": "Expected developments in 1-3 months",
            "medium_term": "Projected changes in 3-12 months",
            "long_term": "Long-term implications beyond 1 year"
        }
        
        # Analyze issue publication patterns
        recent_count = len([issue for issue in issues 
                          if (datetime.now() - issue.published_date).days <= 7])
        
        if recent_count > len(issues) * 0.5:
            timeline["immediate"] = "High activity period with rapid developments"
        
        return timeline
    
    def _calculate_prediction_confidence(self, issues: List[EnhancedIssue]) -> float:
        """Calculate confidence in predictions."""
        if not issues:
            return 0.0
        
        confidence = 0.0
        
        # More issues = higher confidence
        volume_confidence = min(len(issues) / 50, 1.0)
        confidence += volume_confidence * 0.3
        
        # Higher average relevance = higher confidence
        avg_relevance = np.mean([issue.relevance_score for issue in issues])
        confidence += avg_relevance * 0.3
        
        # Diverse sources = higher confidence
        unique_sources = len(set(issue.source.name for issue in issues))
        source_diversity = min(unique_sources / 10, 1.0)
        confidence += source_diversity * 0.2
        
        # Recent data = higher confidence
        recent_issues = len([issue for issue in issues 
                           if (datetime.now() - issue.published_date).days <= 14])
        recency_factor = recent_issues / len(issues)
        confidence += recency_factor * 0.2
        
        return min(confidence, 1.0)


class CrossReferenceValidator:
    """Cross-reference validation system for fact-checking."""
    
    def __init__(self):
        self.fact_patterns = {
            'date': r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b'
        }
    
    async def validate_issues(self, issues: List[EnhancedIssue]) -> List[EnhancedIssue]:
        """Validate issues through cross-referencing."""
        for issue in issues:
            try:
                issue.fact_check_score = await self._calculate_fact_check_score(issue, issues)
                issue.contradiction_flags = await self._find_contradictions(issue, issues)
            except Exception as e:
                logger.error(f"Validation failed for issue: {e}")
                issue.fact_check_score = 0.5  # Neutral score
        
        return issues
    
    async def _calculate_fact_check_score(self, issue: EnhancedIssue, all_issues: List[EnhancedIssue]) -> float:
        """Calculate fact-checking score based on cross-references."""
        score = 0.5  # Start with neutral
        
        # Extract facts from this issue
        facts = self._extract_facts(issue.content)
        
        # Check consistency with other issues
        confirmations = 0
        contradictions = 0
        
        for other_issue in all_issues:
            if other_issue == issue:
                continue
            
            other_facts = self._extract_facts(other_issue.content)
            
            # Simple fact comparison
            for fact_type, fact_value in facts.items():
                if fact_type in other_facts:
                    if fact_value == other_facts[fact_type]:
                        confirmations += 1
                    else:
                        contradictions += 1
        
        # Calculate score based on confirmations vs contradictions
        total_checks = confirmations + contradictions
        if total_checks > 0:
            consistency_ratio = confirmations / total_checks
            score = 0.3 + (consistency_ratio * 0.7)  # Scale to 0.3-1.0
        
        # Boost score for high-tier sources
        if issue.source.tier == 1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_facts(self, content: str) -> Dict[str, str]:
        """Extract verifiable facts from content."""
        import re
        facts = {}
        
        for fact_type, pattern in self.fact_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                facts[fact_type] = matches[0]  # Take first match
        
        return facts
    
    async def _find_contradictions(self, issue: EnhancedIssue, all_issues: List[EnhancedIssue]) -> List[str]:
        """Find contradictions with other issues."""
        contradictions = []
        
        # Simple contradiction detection
        positive_keywords = ['increase', 'growth', 'rise', 'up', 'gain', 'positive']
        negative_keywords = ['decrease', 'decline', 'fall', 'down', 'loss', 'negative']
        
        issue_sentiment = self._detect_direction(issue.content, positive_keywords, negative_keywords)
        
        for other_issue in all_issues:
            if other_issue == issue or other_issue.source.name == issue.source.name:
                continue
            
            other_sentiment = self._detect_direction(other_issue.content, positive_keywords, negative_keywords)
            
            if issue_sentiment and other_sentiment and issue_sentiment != other_sentiment:
                contradictions.append(f"Contradicts {other_issue.source.name} regarding direction/trend")
        
        return contradictions[:3]  # Limit to 3 contradictions
    
    def _detect_direction(self, content: str, positive_keywords: List[str], negative_keywords: List[str]) -> Optional[str]:
        """Detect overall direction sentiment in content."""
        content_lower = content.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        
        return None


class PerformanceOptimizer:
    """Performance optimization and caching system."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 24 * 3600  # 24 hours
    
    def get_cache_key(self, topic: str, timeframe: str) -> str:
        """Generate cache key for search results."""
        return hashlib.md5(f"{topic}:{timeframe}".encode()).hexdigest()
    
    async def get_cached_results(self, topic: str, timeframe: str) -> Optional[List[EnhancedIssue]]:
        """Get cached search results if available and fresh."""
        cache_key = self.get_cache_key(topic, timeframe)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                logger.info(f"Using cached results for {topic}")
                return cached_data['results']
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        return None
    
    async def cache_results(self, topic: str, timeframe: str, results: List[EnhancedIssue]):
        """Cache search results."""
        cache_key = self.get_cache_key(topic, timeframe)
        
        # Convert issues to serializable format for caching
        serializable_results = []
        for issue in results:
            issue_dict = asdict(issue)
            # Remove non-serializable fields
            issue_dict.pop('embedding', None)
            issue_dict.pop('source', None)  # Will need special handling
            serializable_results.append(issue_dict)
        
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'results': results  # Keep original objects for immediate use
        }
        
        logger.info(f"Cached {len(results)} results for {topic}")


class EnhancedIssueSearcher:
    """Main enhanced issue searcher orchestrating all components."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
        
        # Initialize all components
        self.source_managers = {
            'news': NewsSourceManager(),
            'academic': AcademicSourceManager(),
            'social': SocialMediaManager(),
            'technical': TechnicalSourceManager(),
            'financial': FinancialSourceManager()
        }
        
        self.query_generator = AdvancedQueryGenerator(openai_api_key)
        self.deep_miner = DeepWebMiner()
        self.deduplicator = SemanticDeduplicator()
        self.relevance_scorer = EnhancedRelevanceScorer()
        self.sentiment_analyzer = SentimentAnalyzer(openai_api_key)
        self.predictive_analyzer = PredictiveAnalyzer(openai_api_key)
        self.validator = CrossReferenceValidator()
        self.optimizer = PerformanceOptimizer()
        
        logger.info("Enhanced Issue Searcher initialized with all components")
    
    async def comprehensive_search(self, topic: str, timeframe: str = "1주일") -> SearchResult:
        """Perform comprehensive search with all enhancements."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting comprehensive search for: {topic}")
            
            # Check cache first
            cached_results = await self.optimizer.get_cached_results(topic, timeframe)
            if cached_results:
                return await self._convert_to_search_result(cached_results, topic, timeframe, start_time)
            
            # Step 1: Generate search query variations
            all_queries = {}
            for source_type in self.source_managers.keys():
                queries = await self.query_generator.generate_query_variations(topic, source_type)
                all_queries[source_type] = queries
            
            logger.info(f"Generated {sum(len(q) for q in all_queries.values())} total queries")
            
            # Step 2: Parallel multi-source search
            all_issues = await self._parallel_search_all_sources(all_queries)
            logger.info(f"Found {len(all_issues)} raw issues from all sources")
            
            if not all_issues:
                logger.warning("No issues found from any source")
                return self._create_empty_result(topic, timeframe, start_time)
            
            # Step 3: Deep web mining
            enhanced_issues = await self.deep_miner.mine_additional_content(all_issues)
            logger.info("Completed deep web mining")
            
            # Step 4: Deduplication and clustering
            deduplicated_issues = await self.deduplicator.deduplicate_and_cluster(enhanced_issues)
            logger.info(f"Deduplicated to {len(deduplicated_issues)} unique issues")
            
            # Step 5: Enhanced relevance scoring
            scored_issues = await self.relevance_scorer.score_issues(deduplicated_issues, topic)
            logger.info("Completed relevance scoring")
            
            # Step 6: Sentiment and impact analysis
            analyzed_issues = await self.sentiment_analyzer.analyze_sentiment_and_impact(scored_issues)
            logger.info("Completed sentiment and impact analysis")
            
            # Step 7: Cross-reference validation
            validated_issues = await self.validator.validate_issues(analyzed_issues)
            logger.info("Completed cross-reference validation")
            
            # Step 8: Filter and limit results
            final_issues = self._filter_and_limit_results(validated_issues, target_count=75)
            logger.info(f"Final result set: {len(final_issues)} issues")
            
            # Step 9: Generate predictions
            predictions = await self.predictive_analyzer.generate_predictions(final_issues, topic)
            logger.info("Generated predictive analysis")
            
            # Cache results
            await self.optimizer.cache_results(topic, timeframe, final_issues)
            
            # Convert to SearchResult format
            search_result = await self._convert_to_search_result(final_issues, topic, timeframe, start_time)
            search_result.metadata['predictions'] = predictions
            
            total_time = time.time() - start_time
            logger.info(f"Comprehensive search completed in {total_time:.2f}s")
            
            return search_result
            
        except Exception as e:
            logger.error(f"Comprehensive search failed: {e}")
            return self._create_error_result(topic, timeframe, start_time, str(e))
    
    async def _parallel_search_all_sources(self, all_queries: Dict[str, List[str]]) -> List[EnhancedIssue]:
        """Execute parallel searches across all sources."""
        search_tasks = []
        
        for source_type, queries in all_queries.items():
            if source_type in self.source_managers:
                manager = self.source_managers[source_type]
                
                # Limit queries per source to manage performance
                limited_queries = queries[:10]
                
                for query in limited_queries:
                    task = self._search_single_source(manager, query, limit=5)
                    search_tasks.append(task)
        
        # Execute all searches in parallel
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_issues = []
        for result in results:
            if isinstance(result, list):
                all_issues.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
        
        return all_issues
    
    async def _search_single_source(self, manager: SourceManager, query: str, limit: int) -> List[EnhancedIssue]:
        """Search a single source with error handling."""
        try:
            async with manager:
                issues = await manager.search(query, limit)
                return issues
        except Exception as e:
            logger.error(f"Search failed for {manager.source_info.name} with query '{query}': {e}")
            return []
    
    def _filter_and_limit_results(self, issues: List[EnhancedIssue], target_count: int = 75) -> List[EnhancedIssue]:
        """Filter and limit results to target count with quality thresholds."""
        # Filter by minimum quality thresholds
        quality_filtered = []
        
        for issue in issues:
            # Quality checks
            if (issue.relevance_score >= 0.3 and  # Minimum relevance
                len(issue.content) >= 50 and  # Minimum content length
                issue.fact_check_score >= 0.4):  # Minimum fact-check score
                quality_filtered.append(issue)
        
        # Sort by composite score
        def composite_score(issue):
            return (
                issue.relevance_score * 0.4 +
                issue.impact_score * 0.3 +
                issue.fact_check_score * 0.2 +
                issue.authority_score * 0.1
            )
        
        quality_filtered.sort(key=composite_score, reverse=True)
        
        # Return top results up to target count
        return quality_filtered[:target_count]
    
    async def _convert_to_search_result(
        self, 
        enhanced_issues: List[EnhancedIssue], 
        topic: str, 
        timeframe: str, 
        start_time: float
    ) -> SearchResult:
        """Convert enhanced issues to SearchResult format."""
        # Convert enhanced issues to IssueItem format
        issue_items = []
        
        for enhanced_issue in enhanced_issues:
            issue_item = IssueItem(
                title=enhanced_issue.title,
                summary=enhanced_issue.summary,
                source=enhanced_issue.source.name,
                published_date=enhanced_issue.published_date.isoformat() if enhanced_issue.published_date else None,
                relevance_score=enhanced_issue.relevance_score,
                category=enhanced_issue.category,
                content_snippet=enhanced_issue.summary[:200],
                detailed_content=enhanced_issue.content,
                background_context=f"Source: {enhanced_issue.source.name} | Authority: {enhanced_issue.authority_score:.2f}"
            )
            
            # Add enhanced metadata as dynamic attributes
            setattr(issue_item, 'combined_confidence', enhanced_issue.fact_check_score)
            setattr(issue_item, 'sentiment_score', enhanced_issue.sentiment_score)
            setattr(issue_item, 'impact_score', enhanced_issue.impact_score)
            setattr(issue_item, 'controversy_score', enhanced_issue.controversy_score)
            setattr(issue_item, 'momentum_score', enhanced_issue.momentum_score)
            setattr(issue_item, 'authority_score', enhanced_issue.authority_score)
            setattr(issue_item, 'social_engagement', enhanced_issue.shares + enhanced_issue.comments + enhanced_issue.likes)
            setattr(issue_item, 'cross_reference_count', enhanced_issue.cross_reference_count)
            setattr(issue_item, 'entities', enhanced_issue.entities)
            setattr(issue_item, 'tags', enhanced_issue.tags)
            
            issue_items.append(issue_item)
        
        # Generate comprehensive keywords
        all_keywords = [topic]
        for issue in enhanced_issues[:10]:  # Use top 10 issues for keywords
            all_keywords.extend(issue.keywords[:3])  # Top 3 keywords per issue
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(all_keywords))
        
        search_result = SearchResult(
            topic=topic,
            keywords=unique_keywords[:20],  # Limit to 20 keywords
            period=timeframe,
            issues=issue_items,
            total_found=len(issue_items),
            search_time=time.time() - start_time
        )
        
        # Add metadata
        search_result.metadata = {
            'enhanced_search': True,
            'total_sources_searched': len(self.source_managers),
            'average_relevance': np.mean([issue.relevance_score for issue in enhanced_issues]) if enhanced_issues else 0,
            'average_confidence': np.mean([issue.fact_check_score for issue in enhanced_issues]) if enhanced_issues else 0,
            'source_distribution': self._calculate_source_distribution(enhanced_issues),
            'sentiment_analysis': self._calculate_sentiment_summary(enhanced_issues),
            'search_method': 'comprehensive_enhanced'
        }
        
        return search_result
    
    def _calculate_source_distribution(self, issues: List[EnhancedIssue]) -> Dict[str, int]:
        """Calculate distribution of issues across source types."""
        distribution = {}
        for issue in issues:
            source_type = issue.source.type
            distribution[source_type] = distribution.get(source_type, 0) + 1
        return distribution
    
    def _calculate_sentiment_summary(self, issues: List[EnhancedIssue]) -> Dict[str, Any]:
        """Calculate sentiment analysis summary."""
        if not issues:
            return {}
        
        sentiments = [issue.sentiment_score for issue in issues if issue.sentiment_score != 0]
        
        if not sentiments:
            return {'average': 0.0, 'distribution': 'neutral'}
        
        avg_sentiment = np.mean(sentiments)
        
        positive_count = len([s for s in sentiments if s > 0.1])
        negative_count = len([s for s in sentiments if s < -0.1])
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return {
            'average': avg_sentiment,
            'distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'dominant_sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
        }
    
    def _create_empty_result(self, topic: str, timeframe: str, start_time: float) -> SearchResult:
        """Create empty search result."""
        return SearchResult(
            topic=topic,
            keywords=[topic],
            period=timeframe,
            issues=[],
            total_found=0,
            search_time=time.time() - start_time
        )
    
    def _create_error_result(self, topic: str, timeframe: str, start_time: float, error: str) -> SearchResult:
        """Create error search result."""
        result = self._create_empty_result(topic, timeframe, start_time)
        result.metadata = {'error': error, 'enhanced_search': True}
        return result