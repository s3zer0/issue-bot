"""
Enhanced Search Integration Module

Integrates the enhanced search system with the existing bot infrastructure.
Provides seamless transition and backward compatibility while enabling
advanced search capabilities.
"""

import asyncio
import time
from typing import Optional, Union, Dict, Any, List
from datetime import datetime
from loguru import logger

from src.models import SearchResult, IssueItem
from src.search.enhanced_issue_searcher import EnhancedIssueSearcher
from src.search.learning_system import ContinuousLearningOrchestrator
from src.search.issue_searcher import IssueSearcher  # Existing searcher
from src.config import config


class SearchStrategy:
    """Enum-like class for search strategies."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ADAPTIVE = "adaptive"
    EXPERIMENTAL = "experimental"


class SearchModeSelector:
    """Intelligent search mode selection based on various factors."""
    
    def __init__(self, learning_system: Optional[ContinuousLearningOrchestrator] = None):
        self.learning_system = learning_system
        self.mode_weights = {
            SearchStrategy.BASIC: 0.2,
            SearchStrategy.ENHANCED: 0.6,
            SearchStrategy.ADAPTIVE: 0.15,
            SearchStrategy.EXPERIMENTAL: 0.05
        }
    
    async def select_search_strategy(
        self, 
        topic: str, 
        user_id: Optional[str] = None,
        channel_context: Optional[str] = None,
        force_strategy: Optional[str] = None
    ) -> str:
        """Select optimal search strategy based on context."""
        
        if force_strategy:
            return force_strategy
        
        # Analyze topic complexity
        topic_complexity = self._analyze_topic_complexity(topic)
        
        # Check if we have learning data
        has_learning_data = False
        if self.learning_system:
            try:
                insights = await self.learning_system.get_performance_insights(topic)
                has_learning_data = insights.confidence_level > 0.3
            except Exception:
                pass
        
        # Check for A/B test assignment
        if self.learning_system and user_id:
            ab_variant = self.learning_system.ab_test_manager.get_test_variant("search_strategy", user_id)
            if ab_variant:
                return ab_variant.get('search_method', SearchStrategy.ENHANCED)
        
        # Decision logic
        if topic_complexity > 0.8:
            return SearchStrategy.ENHANCED
        elif has_learning_data:
            return SearchStrategy.ADAPTIVE
        elif topic_complexity > 0.5:
            return SearchStrategy.ENHANCED
        else:
            return SearchStrategy.BASIC
    
    def _analyze_topic_complexity(self, topic: str) -> float:
        """Analyze topic complexity to help choose search strategy."""
        complexity_score = 0.0
        
        # Length factor
        word_count = len(topic.split())
        complexity_score += min(word_count / 10, 0.3)
        
        # Technical terms
        technical_terms = ['algorithm', 'framework', 'protocol', 'architecture', 'implementation']
        tech_matches = sum(1 for term in technical_terms if term in topic.lower())
        complexity_score += min(tech_matches / len(technical_terms), 0.3)
        
        # Multi-faceted topics
        if any(word in topic.lower() for word in ['and', 'vs', 'versus', 'comparison', 'analysis']):
            complexity_score += 0.2
        
        # Trending/time-sensitive topics
        if any(word in topic.lower() for word in ['latest', 'new', 'recent', '2024', 'breakthrough']):
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)


class HybridSearchOrchestrator:
    """Main orchestrator that manages both basic and enhanced search systems."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or config.get_openai_api_key()
        
        # Initialize search systems
        self.basic_searcher = IssueSearcher()
        self.enhanced_searcher = EnhancedIssueSearcher(openai_api_key)
        self.learning_system = ContinuousLearningOrchestrator()
        self.mode_selector = SearchModeSelector(self.learning_system)
        
        # Performance tracking
        self.search_stats = {
            'basic': {'count': 0, 'total_time': 0, 'avg_results': 0},
            'enhanced': {'count': 0, 'total_time': 0, 'avg_results': 0},
            'adaptive': {'count': 0, 'total_time': 0, 'avg_results': 0}
        }
        
        logger.info("Hybrid Search Orchestrator initialized")
    
    async def search_issues(
        self,
        topic: str,
        timeframe: str = "1주일",
        user_id: Optional[str] = None,
        channel_context: Optional[str] = None,
        force_strategy: Optional[str] = None,
        feedback_callback: Optional[callable] = None
    ) -> SearchResult:
        """
        Main search method that intelligently selects and executes search strategy.
        
        Args:
            topic: Search topic
            timeframe: Time period for search
            user_id: User ID for personalization and A/B testing
            channel_context: Discord channel context for optimization
            force_strategy: Force specific strategy (basic/enhanced/adaptive)
            feedback_callback: Callback function for real-time feedback
            
        Returns:
            SearchResult with enhanced metadata
        """
        start_time = time.time()
        
        try:
            # Select search strategy
            strategy = await self.mode_selector.select_search_strategy(
                topic, user_id, channel_context, force_strategy
            )
            
            logger.info(f"Selected search strategy: {strategy} for topic: {topic}")
            
            if feedback_callback:
                await feedback_callback(f"🔍 Searching with {strategy} strategy...")
            
            # Execute search based on strategy
            if strategy == SearchStrategy.BASIC:
                result = await self._execute_basic_search(topic, timeframe)
            elif strategy == SearchStrategy.ENHANCED:
                result = await self._execute_enhanced_search(topic, timeframe)
            elif strategy == SearchStrategy.ADAPTIVE:
                result = await self._execute_adaptive_search(topic, timeframe, user_id)
            else:  # EXPERIMENTAL
                result = await self._execute_experimental_search(topic, timeframe)
            
            # Add strategy metadata
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata['search_strategy'] = strategy
            result.metadata['search_orchestrator'] = 'hybrid'
            
            # Record performance metrics
            search_time = time.time() - start_time
            await self._record_search_performance(strategy, result, search_time)
            
            # Record for learning system
            if user_id:
                await self.learning_system.record_search_session(
                    topic=topic,
                    search_method=strategy,
                    total_found=result.total_found,
                    engaged_count=0,  # Will be updated later with user interactions
                    avg_relevance=self._calculate_avg_relevance(result),
                    search_time=search_time
                )
            
            logger.info(f"Search completed: {result.total_found} issues found in {search_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Search failed for topic '{topic}': {e}")
            # Fallback to basic search
            try:
                result = await self._execute_basic_search(topic, timeframe)
                result.metadata = {'error': str(e), 'fallback': True, 'search_strategy': 'basic'}
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return self._create_empty_result(topic, timeframe, time.time() - start_time)
    
    async def _execute_basic_search(self, topic: str, timeframe: str) -> SearchResult:
        """Execute basic search using existing system."""
        try:
            # Use existing issue searcher
            result = await self.basic_searcher.search_issues(topic, timeframe)
            
            # Enhance with minimal metadata
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata['search_method'] = 'basic'
            result.metadata['enhanced'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            raise
    
    async def _execute_enhanced_search(self, topic: str, timeframe: str) -> SearchResult:
        """Execute enhanced search with all advanced features."""
        try:
            result = await self.enhanced_searcher.comprehensive_search(topic, timeframe)
            return result
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Fallback to basic
            return await self._execute_basic_search(topic, timeframe)
    
    async def _execute_adaptive_search(self, topic: str, timeframe: str, user_id: Optional[str]) -> SearchResult:
        """Execute adaptive search using learning-optimized parameters."""
        try:
            # Get optimized parameters from learning system
            optimized_config = await self.learning_system.get_optimized_search_config(topic)
            
            # Apply optimizations to enhanced searcher
            await self._apply_search_optimizations(optimized_config)
            
            # Execute enhanced search with optimized parameters
            result = await self.enhanced_searcher.comprehensive_search(topic, timeframe)
            
            # Add adaptive metadata
            result.metadata['adaptive_optimizations'] = optimized_config
            result.metadata['search_method'] = 'adaptive'
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive search failed: {e}")
            # Fallback to enhanced search
            return await self._execute_enhanced_search(topic, timeframe)
    
    async def _execute_experimental_search(self, topic: str, timeframe: str) -> SearchResult:
        """Execute experimental search strategies."""
        try:
            # For now, use enhanced search with experimental parameters
            # In the future, this could test new algorithms
            result = await self.enhanced_searcher.comprehensive_search(topic, timeframe)
            result.metadata['search_method'] = 'experimental'
            result.metadata['experimental_features'] = ['new_relevance_scoring', 'beta_clustering']
            
            return result
            
        except Exception as e:
            logger.error(f"Experimental search failed: {e}")
            return await self._execute_enhanced_search(topic, timeframe)
    
    async def _apply_search_optimizations(self, config: Dict[str, Any]):
        """Apply optimized configuration to enhanced searcher."""
        try:
            # Adjust source weights
            if 'source_weights' in config:
                # This would require modifying the enhanced searcher's source managers
                # For now, log the optimization
                logger.info(f"Applying source weight optimizations: {config['source_weights']}")
            
            # Adjust relevance threshold
            if 'relevance_threshold' in config:
                # This would require modifying the relevance scorer
                logger.info(f"Applying relevance threshold: {config['relevance_threshold']}")
            
            # Other optimizations...
            
        except Exception as e:
            logger.error(f"Failed to apply search optimizations: {e}")
    
    async def _record_search_performance(self, strategy: str, result: SearchResult, search_time: float):
        """Record search performance for analysis."""
        if strategy not in self.search_stats:
            self.search_stats[strategy] = {'count': 0, 'total_time': 0, 'avg_results': 0}
        
        stats = self.search_stats[strategy]
        stats['count'] += 1
        stats['total_time'] += search_time
        stats['avg_results'] = (stats['avg_results'] * (stats['count'] - 1) + result.total_found) / stats['count']
    
    def _calculate_avg_relevance(self, result: SearchResult) -> float:
        """Calculate average relevance score from search result."""
        if not result.issues:
            return 0.0
        
        total_relevance = sum(issue.relevance_score for issue in result.issues)
        return total_relevance / len(result.issues)
    
    def _create_empty_result(self, topic: str, timeframe: str, search_time: float) -> SearchResult:
        """Create empty search result for error cases."""
        return SearchResult(
            topic=topic,
            keywords=[topic],
            period=timeframe,
            issues=[],
            total_found=0,
            search_time=search_time
        )
    
    async def record_user_interaction(
        self,
        topic: str,
        issue_title: str,
        interaction_type: str,
        user_id: Optional[str] = None,
        engagement_time: Optional[float] = None
    ):
        """Record user interaction for learning system."""
        try:
            await self.learning_system.record_user_interaction(
                topic=topic,
                issue_title=issue_title,
                interaction_type=interaction_type,
                user_id=user_id,
                engagement_time=engagement_time
            )
        except Exception as e:
            logger.error(f"Failed to record user interaction: {e}")
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search performance analytics."""
        analytics = {
            'performance_stats': self.search_stats.copy(),
            'strategy_recommendations': await self._generate_strategy_recommendations()
        }
        
        # Calculate overall metrics
        total_searches = sum(stats['count'] for stats in self.search_stats.values())
        if total_searches > 0:
            analytics['overall'] = {
                'total_searches': total_searches,
                'avg_search_time': sum(stats['total_time'] for stats in self.search_stats.values()) / total_searches,
                'avg_results_per_search': sum(stats['avg_results'] * stats['count'] for stats in self.search_stats.values()) / total_searches
            }
        
        return analytics
    
    async def _generate_strategy_recommendations(self) -> List[str]:
        """Generate recommendations for strategy optimization."""
        recommendations = []
        
        # Analyze performance differences
        if (self.search_stats['enhanced']['count'] > 5 and 
            self.search_stats['basic']['count'] > 5):
            
            enhanced_avg_results = self.search_stats['enhanced']['avg_results']
            basic_avg_results = self.search_stats['basic']['avg_results']
            
            if enhanced_avg_results > basic_avg_results * 2:
                recommendations.append("Enhanced search significantly outperforms basic - increase enhanced usage")
            
            enhanced_avg_time = (self.search_stats['enhanced']['total_time'] / 
                               self.search_stats['enhanced']['count'])
            basic_avg_time = (self.search_stats['basic']['total_time'] / 
                            self.search_stats['basic']['count'])
            
            if enhanced_avg_time > basic_avg_time * 3:
                recommendations.append("Enhanced search is slow - consider performance optimizations")
        
        return recommendations


# Convenience functions for easy integration
async def search_issues_intelligent(
    topic: str,
    timeframe: str = "1주일",
    user_id: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    force_strategy: Optional[str] = None
) -> SearchResult:
    """
    Convenience function for intelligent issue searching.
    
    Automatically selects the best search strategy and executes it.
    """
    orchestrator = HybridSearchOrchestrator(openai_api_key)
    return await orchestrator.search_issues(
        topic=topic,
        timeframe=timeframe,
        user_id=user_id,
        force_strategy=force_strategy
    )


async def search_issues_enhanced_only(
    topic: str,
    timeframe: str = "1주일",
    openai_api_key: Optional[str] = None
) -> SearchResult:
    """
    Convenience function for enhanced search only.
    
    Always uses the enhanced search system with all advanced features.
    """
    searcher = EnhancedIssueSearcher(openai_api_key)
    return await searcher.comprehensive_search(topic, timeframe)


class SearchResultComparator:
    """Compare results between different search strategies."""
    
    @staticmethod
    async def compare_search_strategies(
        topic: str,
        timeframe: str = "1주일",
        openai_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare basic vs enhanced search for a topic."""
        orchestrator = HybridSearchOrchestrator(openai_api_key)
        
        # Run both searches
        basic_result = await orchestrator._execute_basic_search(topic, timeframe)
        enhanced_result = await orchestrator._execute_enhanced_search(topic, timeframe)
        
        comparison = {
            'topic': topic,
            'basic': {
                'total_found': basic_result.total_found,
                'search_time': basic_result.search_time,
                'avg_relevance': orchestrator._calculate_avg_relevance(basic_result)
            },
            'enhanced': {
                'total_found': enhanced_result.total_found,
                'search_time': enhanced_result.search_time,
                'avg_relevance': orchestrator._calculate_avg_relevance(enhanced_result),
                'metadata': enhanced_result.metadata
            }
        }
        
        # Calculate improvement metrics
        if basic_result.total_found > 0:
            comparison['improvement'] = {
                'results_multiplier': enhanced_result.total_found / basic_result.total_found,
                'time_ratio': enhanced_result.search_time / basic_result.search_time,
                'quality_improvement': (comparison['enhanced']['avg_relevance'] - 
                                      comparison['basic']['avg_relevance'])
            }
        
        return comparison


# Integration with existing bot commands
class EnhancedSearchMixin:
    """Mixin to add enhanced search capabilities to existing bot classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_orchestrator = HybridSearchOrchestrator()
    
    async def enhanced_monitor_command(
        self,
        ctx,
        topic: str,
        period: str = "1주일",
        strategy: Optional[str] = None
    ):
        """Enhanced version of monitor command with intelligent search selection."""
        try:
            user_id = str(ctx.author.id) if hasattr(ctx, 'author') else None
            
            # Send initial response
            await ctx.send(f"🔍 Analyzing '{topic}' with intelligent search...")
            
            # Perform search
            result = await self.search_orchestrator.search_issues(
                topic=topic,
                timeframe=period,
                user_id=user_id,
                force_strategy=strategy
            )
            
            # Send results using existing reporting system
            # (This would integrate with your existing report generation)
            
            # Record interaction for learning
            if result.issues:
                await self.search_orchestrator.record_user_interaction(
                    topic=topic,
                    issue_title=result.issues[0].title,
                    interaction_type="view",
                    user_id=user_id
                )
            
        except Exception as e:
            await ctx.send(f"❌ Search failed: {str(e)}")
    
    async def search_analytics_command(self, ctx):
        """Command to show search analytics."""
        try:
            analytics = await self.search_orchestrator.get_search_analytics()
            
            # Format analytics for Discord
            # (Implementation depends on your Discord formatting preferences)
            
        except Exception as e:
            await ctx.send(f"❌ Analytics retrieval failed: {str(e)}")


# Real-time RSS monitoring (placeholder for future implementation)
class RealTimeMonitor:
    """Real-time RSS and webhook monitoring for breaking news."""
    
    def __init__(self, orchestrator: HybridSearchOrchestrator):
        self.orchestrator = orchestrator
        self.monitored_topics = set()
        self.rss_feeds = []
    
    async def add_topic_monitoring(self, topic: str):
        """Add a topic for real-time monitoring."""
        self.monitored_topics.add(topic)
        logger.info(f"Added real-time monitoring for: {topic}")
    
    async def start_monitoring(self):
        """Start real-time monitoring (placeholder)."""
        # Implementation would include:
        # - RSS feed polling
        # - Webhook listeners
        # - Real-time alerts
        pass