"""
Continuous Learning System

Tracks user engagement and feedback to improve search quality over time.
Implements A/B testing, feedback collection, and adaptive parameter tuning.
"""

import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from loguru import logger

from src.search.enhanced_search_engine import EnhancedIssue
from src.models import SearchResult


@dataclass
class UserFeedback:
    """User feedback on search results."""
    timestamp: datetime
    topic: str
    issue_id: str
    issue_title: str
    feedback_type: str  # 'like', 'dislike', 'irrelevant', 'helpful'
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    engagement_time: Optional[float] = None  # Time spent viewing
    additional_notes: Optional[str] = None


@dataclass
class SearchMetrics:
    """Metrics for a search session."""
    timestamp: datetime
    topic: str
    search_method: str  # 'basic', 'enhanced', 'experimental'
    total_issues_found: int
    issues_engaged_with: int
    average_relevance_score: float
    search_time: float
    user_satisfaction: Optional[float] = None  # 1-5 scale
    success_indicators: List[str] = None


@dataclass
class LearningInsights:
    """Insights derived from learning analysis."""
    topic_category: str
    optimal_source_weights: Dict[str, float]
    best_performing_queries: List[str]
    user_preference_patterns: Dict[str, Any]
    recommended_adjustments: List[str]
    confidence_level: float


class FeedbackCollector:
    """Collects and stores user feedback."""
    
    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                topic TEXT NOT NULL,
                issue_id TEXT NOT NULL,
                issue_title TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                user_id TEXT,
                channel_id TEXT,
                engagement_time REAL,
                additional_notes TEXT
            )
        ''')
        
        # Search metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                topic TEXT NOT NULL,
                search_method TEXT NOT NULL,
                total_issues_found INTEGER,
                issues_engaged_with INTEGER,
                average_relevance_score REAL,
                search_time REAL,
                user_satisfaction REAL,
                success_indicators TEXT
            )
        ''')
        
        # Source performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                topic_category TEXT NOT NULL,
                relevance_score REAL,
                engagement_rate REAL,
                accuracy_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def record_feedback(self, feedback: UserFeedback):
        """Record user feedback."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (
                    timestamp, topic, issue_id, issue_title, feedback_type,
                    user_id, channel_id, engagement_time, additional_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.timestamp.isoformat(),
                feedback.topic,
                feedback.issue_id,
                feedback.issue_title,
                feedback.feedback_type,
                feedback.user_id,
                feedback.channel_id,
                feedback.engagement_time,
                feedback.additional_notes
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded feedback: {feedback.feedback_type} for {feedback.issue_title}")
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    async def record_search_metrics(self, metrics: SearchMetrics):
        """Record search session metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            success_indicators_json = json.dumps(metrics.success_indicators or [])
            
            cursor.execute('''
                INSERT INTO search_metrics (
                    timestamp, topic, search_method, total_issues_found,
                    issues_engaged_with, average_relevance_score, search_time,
                    user_satisfaction, success_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.topic,
                metrics.search_method,
                metrics.total_issues_found,
                metrics.issues_engaged_with,
                metrics.average_relevance_score,
                metrics.search_time,
                metrics.user_satisfaction,
                success_indicators_json
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded search metrics for topic: {metrics.topic}")
            
        except Exception as e:
            logger.error(f"Failed to record search metrics: {e}")
    
    async def get_feedback_for_topic(self, topic: str, days_back: int = 30) -> List[UserFeedback]:
        """Get feedback for a specific topic."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            cursor.execute('''
                SELECT * FROM feedback 
                WHERE topic = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (topic, cutoff_date))
            
            rows = cursor.fetchall()
            conn.close()
            
            feedback_list = []
            for row in rows:
                feedback = UserFeedback(
                    timestamp=datetime.fromisoformat(row[1]),
                    topic=row[2],
                    issue_id=row[3],
                    issue_title=row[4],
                    feedback_type=row[5],
                    user_id=row[6],
                    channel_id=row[7],
                    engagement_time=row[8],
                    additional_notes=row[9]
                )
                feedback_list.append(feedback)
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"Failed to get feedback for topic {topic}: {e}")
            return []


class PerformanceAnalyzer:
    """Analyzes search performance and generates insights."""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
    
    async def analyze_topic_performance(self, topic: str, days_back: int = 30) -> LearningInsights:
        """Analyze performance for a specific topic."""
        try:
            # Get feedback data
            feedback_data = await self.feedback_collector.get_feedback_for_topic(topic, days_back)
            
            # Analyze source performance
            source_weights = await self._analyze_source_performance(feedback_data)
            
            # Identify best queries
            best_queries = await self._identify_best_queries(topic, feedback_data)
            
            # Analyze user preferences
            user_patterns = await self._analyze_user_patterns(feedback_data)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(feedback_data, source_weights)
            
            # Calculate confidence
            confidence = self._calculate_insight_confidence(feedback_data)
            
            return LearningInsights(
                topic_category=await self._categorize_topic(topic),
                optimal_source_weights=source_weights,
                best_performing_queries=best_queries,
                user_preference_patterns=user_patterns,
                recommended_adjustments=recommendations,
                confidence_level=confidence
            )
            
        except Exception as e:
            logger.error(f"Performance analysis failed for {topic}: {e}")
            return self._default_insights(topic)
    
    async def _analyze_source_performance(self, feedback_data: List[UserFeedback]) -> Dict[str, float]:
        """Analyze which sources perform best based on feedback."""
        source_scores = {}
        source_counts = {}
        
        for feedback in feedback_data:
            # Extract source from issue_id (simplified)
            source = self._extract_source_from_feedback(feedback)
            
            if source not in source_scores:
                source_scores[source] = 0.0
                source_counts[source] = 0
            
            # Score based on feedback type
            score = self._feedback_to_score(feedback.feedback_type)
            source_scores[source] += score
            source_counts[source] += 1
        
        # Calculate average scores and normalize to weights
        weights = {}
        total_score = 0
        
        for source, total_score_source in source_scores.items():
            if source_counts[source] > 0:
                avg_score = total_score_source / source_counts[source]
                weights[source] = max(avg_score, 0.1)  # Minimum weight of 0.1
                total_score += weights[source]
        
        # Normalize weights to sum to 1.0
        if total_score > 0:
            for source in weights:
                weights[source] /= total_score
        
        return weights
    
    def _extract_source_from_feedback(self, feedback: UserFeedback) -> str:
        """Extract source name from feedback (simplified implementation)."""
        # In a real implementation, this would parse the issue_id
        # or maintain a lookup table
        if "github" in feedback.issue_title.lower():
            return "technical"
        elif "news" in feedback.issue_title.lower():
            return "news"
        elif "research" in feedback.issue_title.lower():
            return "academic"
        elif "twitter" in feedback.issue_title.lower() or "social" in feedback.issue_title.lower():
            return "social"
        else:
            return "general"
    
    def _feedback_to_score(self, feedback_type: str) -> float:
        """Convert feedback type to numerical score."""
        scores = {
            'like': 1.0,
            'helpful': 1.0,
            'dislike': -0.5,
            'irrelevant': -1.0
        }
        return scores.get(feedback_type, 0.0)
    
    async def _identify_best_queries(self, topic: str, feedback_data: List[UserFeedback]) -> List[str]:
        """Identify best performing query patterns."""
        # Simplified implementation - would need query tracking in real system
        return [
            f"{topic} latest",
            f"{topic} news",
            f"{topic} development",
            f"recent {topic}",
            f"{topic} analysis"
        ]
    
    async def _analyze_user_patterns(self, feedback_data: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze user preference patterns."""
        patterns = {
            'preferred_feedback_types': {},
            'engagement_time_avg': 0.0,
            'most_active_hours': [],
            'topic_preferences': {}
        }
        
        # Analyze feedback types
        for feedback in feedback_data:
            ftype = feedback.feedback_type
            patterns['preferred_feedback_types'][ftype] = patterns['preferred_feedback_types'].get(ftype, 0) + 1
        
        # Analyze engagement times
        engagement_times = [f.engagement_time for f in feedback_data if f.engagement_time]
        if engagement_times:
            patterns['engagement_time_avg'] = np.mean(engagement_times)
        
        return patterns
    
    async def _generate_recommendations(
        self, 
        feedback_data: List[UserFeedback], 
        source_weights: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving search."""
        recommendations = []
        
        # Analyze feedback balance
        positive_feedback = len([f for f in feedback_data if f.feedback_type in ['like', 'helpful']])
        negative_feedback = len([f for f in feedback_data if f.feedback_type in ['dislike', 'irrelevant']])
        
        if negative_feedback > positive_feedback:
            recommendations.append("Increase relevance threshold to filter lower-quality results")
            recommendations.append("Adjust query generation to be more specific")
        
        # Source recommendations
        if source_weights:
            best_source = max(source_weights, key=source_weights.get)
            worst_source = min(source_weights, key=source_weights.get)
            
            if source_weights[best_source] > 0.4:
                recommendations.append(f"Increase weight for {best_source} sources")
            
            if source_weights[worst_source] < 0.1:
                recommendations.append(f"Consider reducing {worst_source} source priority")
        
        # Engagement recommendations
        avg_engagement = np.mean([f.engagement_time for f in feedback_data if f.engagement_time] or [0])
        if avg_engagement < 30:  # Less than 30 seconds
            recommendations.append("Improve issue summaries to increase engagement")
            recommendations.append("Consider adding more compelling titles")
        
        return recommendations
    
    def _calculate_insight_confidence(self, feedback_data: List[UserFeedback]) -> float:
        """Calculate confidence level in insights based on data quality."""
        if not feedback_data:
            return 0.0
        
        # Factors affecting confidence
        data_size_factor = min(len(feedback_data) / 50, 1.0)  # Ideal: 50+ feedback items
        
        # Diversity of feedback types
        feedback_types = set(f.feedback_type for f in feedback_data)
        diversity_factor = min(len(feedback_types) / 4, 1.0)  # Ideal: 4 types
        
        # Recency of data
        recent_feedback = len([f for f in feedback_data 
                             if (datetime.now() - f.timestamp).days <= 7])
        recency_factor = min(recent_feedback / len(feedback_data), 1.0)
        
        confidence = (data_size_factor * 0.4 + diversity_factor * 0.3 + recency_factor * 0.3)
        return confidence
    
    async def _categorize_topic(self, topic: str) -> str:
        """Categorize topic for targeted analysis."""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ['tech', 'ai', 'software', 'programming']):
            return "technology"
        elif any(word in topic_lower for word in ['business', 'market', 'economy', 'financial']):
            return "business"
        elif any(word in topic_lower for word in ['research', 'study', 'science']):
            return "academic"
        else:
            return "general"
    
    def _default_insights(self, topic: str) -> LearningInsights:
        """Generate default insights when analysis fails."""
        return LearningInsights(
            topic_category="general",
            optimal_source_weights={
                "news": 0.3,
                "technical": 0.25,
                "academic": 0.2,
                "social": 0.15,
                "financial": 0.1
            },
            best_performing_queries=[topic, f"{topic} latest", f"{topic} news"],
            user_preference_patterns={},
            recommended_adjustments=["Collect more feedback data"],
            confidence_level=0.2
        )


class AdaptiveParameterTuner:
    """Automatically tunes search parameters based on learning insights."""
    
    def __init__(self, feedback_collector: FeedbackCollector, analyzer: PerformanceAnalyzer):
        self.feedback_collector = feedback_collector
        self.analyzer = analyzer
        self.parameter_history = {}
    
    async def get_optimized_parameters(self, topic: str) -> Dict[str, Any]:
        """Get optimized parameters for a topic based on learning."""
        try:
            insights = await self.analyzer.analyze_topic_performance(topic)
            
            if insights.confidence_level < 0.3:
                # Not enough data, use defaults
                return self._get_default_parameters()
            
            # Build optimized parameters
            optimized = {
                'source_weights': insights.optimal_source_weights,
                'relevance_threshold': self._calculate_optimal_threshold(insights),
                'max_results_per_source': self._calculate_optimal_batch_size(insights),
                'query_variations': insights.best_performing_queries,
                'priority_adjustments': self._generate_priority_adjustments(insights)
            }
            
            # Store parameter history
            self.parameter_history[topic] = {
                'timestamp': datetime.now(),
                'parameters': optimized,
                'confidence': insights.confidence_level
            }
            
            logger.info(f"Generated optimized parameters for {topic} (confidence: {insights.confidence_level:.2f})")
            return optimized
            
        except Exception as e:
            logger.error(f"Parameter optimization failed for {topic}: {e}")
            return self._get_default_parameters()
    
    def _calculate_optimal_threshold(self, insights: LearningInsights) -> float:
        """Calculate optimal relevance threshold."""
        # Start with default threshold
        threshold = 0.3
        
        # Adjust based on user feedback patterns
        user_patterns = insights.user_preference_patterns
        negative_feedback_rate = user_patterns.get('preferred_feedback_types', {}).get('irrelevant', 0)
        
        if negative_feedback_rate > 5:  # Too many irrelevant results
            threshold += 0.1
        elif negative_feedback_rate == 0:  # No irrelevant feedback, can be more lenient
            threshold -= 0.05
        
        return max(0.1, min(0.7, threshold))  # Keep in reasonable range
    
    def _calculate_optimal_batch_size(self, insights: LearningInsights) -> int:
        """Calculate optimal number of results per source."""
        base_size = 5
        
        # Adjust based on engagement patterns
        user_patterns = insights.user_preference_patterns
        avg_engagement = user_patterns.get('engagement_time_avg', 0)
        
        if avg_engagement > 60:  # High engagement, users want more content
            return min(base_size + 3, 10)
        elif avg_engagement < 20:  # Low engagement, reduce volume
            return max(base_size - 2, 2)
        
        return base_size
    
    def _generate_priority_adjustments(self, insights: LearningInsights) -> Dict[str, float]:
        """Generate priority adjustments for different content types."""
        adjustments = {}
        
        # Boost sources with good performance
        for source, weight in insights.optimal_source_weights.items():
            if weight > 0.3:
                adjustments[f"{source}_boost"] = 1.2
            elif weight < 0.1:
                adjustments[f"{source}_penalty"] = 0.8
        
        return adjustments
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters when no learning data available."""
        return {
            'source_weights': {
                'news': 0.25,
                'academic': 0.2,
                'technical': 0.2,
                'social': 0.2,
                'financial': 0.15
            },
            'relevance_threshold': 0.3,
            'max_results_per_source': 5,
            'query_variations': [],
            'priority_adjustments': {}
        }


class ABTestManager:
    """Manages A/B testing of different search strategies."""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.active_tests = {}
        self.test_results = {}
    
    async def start_ab_test(self, test_name: str, variants: Dict[str, Dict[str, Any]], duration_days: int = 7):
        """Start a new A/B test."""
        test_config = {
            'name': test_name,
            'variants': variants,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=duration_days),
            'assignment_counter': 0
        }
        
        self.active_tests[test_name] = test_config
        logger.info(f"Started A/B test: {test_name} with {len(variants)} variants")
    
    def get_test_variant(self, test_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get test variant for a user."""
        if test_name not in self.active_tests:
            return None
        
        test = self.active_tests[test_name]
        
        # Check if test is still active
        if datetime.now() > test['end_date']:
            return None
        
        # Simple hash-based assignment for consistency
        user_hash = hash(user_id) % 100
        variant_names = list(test['variants'].keys())
        variant_index = user_hash % len(variant_names)
        
        selected_variant = variant_names[variant_index]
        return test['variants'][selected_variant]
    
    async def record_test_result(self, test_name: str, user_id: str, variant: str, outcome_metrics: Dict[str, Any]):
        """Record A/B test result."""
        if test_name not in self.test_results:
            self.test_results[test_name] = {}
        
        if variant not in self.test_results[test_name]:
            self.test_results[test_name][variant] = []
        
        result = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'metrics': outcome_metrics
        }
        
        self.test_results[test_name][variant].append(result)
        logger.info(f"Recorded A/B test result for {test_name}/{variant}")
    
    async def analyze_test_results(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_name not in self.test_results:
            return {'error': 'No results found'}
        
        variants_data = self.test_results[test_name]
        analysis = {}
        
        for variant, results in variants_data.items():
            if not results:
                continue
            
            # Calculate average metrics
            metrics_summary = {}
            if results:
                first_result = results[0]['metrics']
                for metric_name in first_result.keys():
                    values = [r['metrics'].get(metric_name, 0) for r in results]
                    metrics_summary[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            analysis[variant] = {
                'sample_size': len(results),
                'metrics': metrics_summary
            }
        
        # Determine winner
        if len(analysis) >= 2:
            # Simple winner selection based on primary metric (assuming 'user_satisfaction')
            winner = max(analysis.keys(), 
                        key=lambda v: analysis[v]['metrics'].get('user_satisfaction', {}).get('mean', 0))
            analysis['winner'] = winner
        
        return analysis


class ContinuousLearningOrchestrator:
    """Main orchestrator for the continuous learning system."""
    
    def __init__(self, db_path: str = "data/learning.db"):
        self.feedback_collector = FeedbackCollector(db_path)
        self.analyzer = PerformanceAnalyzer(self.feedback_collector)
        self.parameter_tuner = AdaptiveParameterTuner(self.feedback_collector, self.analyzer)
        self.ab_test_manager = ABTestManager(self.feedback_collector)
        
        logger.info("Continuous Learning System initialized")
    
    async def record_user_interaction(
        self, 
        topic: str, 
        issue_title: str, 
        interaction_type: str,
        user_id: Optional[str] = None,
        engagement_time: Optional[float] = None
    ):
        """Record a user interaction for learning."""
        feedback = UserFeedback(
            timestamp=datetime.now(),
            topic=topic,
            issue_id=hashlib.md5(issue_title.encode()).hexdigest()[:8],
            issue_title=issue_title,
            feedback_type=interaction_type,
            user_id=user_id,
            engagement_time=engagement_time
        )
        
        await self.feedback_collector.record_feedback(feedback)
    
    async def record_search_session(
        self,
        topic: str,
        search_method: str,
        total_found: int,
        engaged_count: int,
        avg_relevance: float,
        search_time: float
    ):
        """Record a complete search session."""
        metrics = SearchMetrics(
            timestamp=datetime.now(),
            topic=topic,
            search_method=search_method,
            total_issues_found=total_found,
            issues_engaged_with=engaged_count,
            average_relevance_score=avg_relevance,
            search_time=search_time
        )
        
        await self.feedback_collector.record_search_metrics(metrics)
    
    async def get_optimized_search_config(self, topic: str) -> Dict[str, Any]:
        """Get optimized search configuration for a topic."""
        return await self.parameter_tuner.get_optimized_parameters(topic)
    
    async def get_performance_insights(self, topic: str) -> LearningInsights:
        """Get performance insights for a topic."""
        return await self.analyzer.analyze_topic_performance(topic)
    
    async def setup_search_ab_test(self, topic_category: str):
        """Setup A/B test for search strategies."""
        variants = {
            'enhanced': {
                'search_method': 'enhanced',
                'source_diversity': 'high',
                'query_expansion': True
            },
            'focused': {
                'search_method': 'focused',
                'source_diversity': 'medium',
                'query_expansion': False
            },
            'balanced': {
                'search_method': 'balanced',
                'source_diversity': 'high',
                'query_expansion': True
            }
        }
        
        await self.ab_test_manager.start_ab_test(
            f"search_strategy_{topic_category}",
            variants,
            duration_days=14
        )
        
        logger.info(f"Started A/B test for {topic_category} search strategies")