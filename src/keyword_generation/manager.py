"""
멀티 소스 키워드 생성 매니저.
"""

from typing import List, Dict, Optional, Set
import asyncio
import time
import json
from dataclasses import dataclass, field
from loguru import logger

from .base import BaseKeywordExtractor, KeywordExtractionResult, KeywordItem, KeywordImportance
from .similarity import KeywordSimilarityAnalyzer
from .clusterer import KeywordClusterer, ClusteringResult
from src.models import KeywordResult  # 기존 모델과의 호환성


@dataclass
class MultiSourceKeywordResult:
    """멀티 소스 키워드 생성 결과."""
    keywords: List[KeywordItem]
    source_results: Dict[str, KeywordExtractionResult]
    total_time: float
    merged_count: int  # 중복으로 병합된 키워드 수
    high_importance_count: int
    normal_importance_count: int
    low_importance_count: int
    expansion_keywords: List[KeywordItem] = field(default_factory=list)  # 확장 키워드
    clustering_result: Optional[ClusteringResult] = None  # 클러스터링 결과
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_legacy_format(self, topic: str) -> KeywordResult:
        """기존 KeywordResult 형식으로 변환 (호환성)."""
        # 카테고리별로 키워드 분류 (확장 키워드 포함)
        primary = []
        related = []
        context = []

        # 원본 키워드 분류
        for kw_item in self.keywords:
            if kw_item.category == 'primary' or kw_item.importance == KeywordImportance.HIGH:
                primary.append(kw_item.keyword)
            elif kw_item.category == 'related':
                related.append(kw_item.keyword)
            else:
                context.append(kw_item.keyword)

        # 확장 키워드를 related_terms에 추가
        expansion_terms = [kw.keyword for kw in self.expansion_keywords]
        related.extend(expansion_terms)

        # 전체 신뢰도 계산 (평균) - 확장 키워드 포함
        all_keywords = self.keywords + self.expansion_keywords
        avg_confidence = sum(kw.confidence for kw in all_keywords) / len(all_keywords) if all_keywords else 0.5

        all_trusted_domains = set()
        for result in self.source_results.values():
            if result.is_success and result.raw_response:
                try:
                    # raw_response가 JSON 형식이라고 가정하고 파싱
                    response_data = json.loads(result.raw_response)
                    domains = response_data.get("trusted_domains", [])
                    if isinstance(domains, list):
                        all_trusted_domains.update(d.lower() for d in domains)
                except (json.JSONDecodeError, TypeError):
                    # JSON 파싱 실패 시 무시
                    continue

        # 확장 키워드 정보를 메타데이터에 포함
        enhanced_metadata = dict(self.metadata)
        enhanced_metadata['expansion_keywords'] = expansion_terms
        if self.clustering_result:
            enhanced_metadata['clustering_summary'] = {
                'num_clusters': len(self.clustering_result.clusters),
                'num_expansion_keywords': len(self.clustering_result.expansion_keywords),
                'clustering_time': self.clustering_result.clustering_time
            }

        return KeywordResult(
            topic=topic,
            primary_keywords=primary[:10],  # 최대 10개
            related_terms=related[:15],  # 확장 키워드 포함으로 더 많이
            context_keywords=context[:10],
            confidence_score=avg_confidence,
            generation_time=self.total_time,
            raw_response=str(enhanced_metadata),
            trusted_domains = list(all_trusted_domains)
        )


class MultiSourceKeywordManager:
    """여러 소스를 통합하여 키워드를 생성하는 매니저."""

    def __init__(
        self,
        extractors: Optional[List[BaseKeywordExtractor]] = None,
        similarity_analyzer: Optional[KeywordSimilarityAnalyzer] = None,
        clusterer: Optional[KeywordClusterer] = None,
        enable_clustering: bool = True
    ):
        """
        멀티 소스 키워드 매니저 초기화.

        Args:
            extractors: 사용할 키워드 추출기 리스트
            similarity_analyzer: 유사도 분석기
            clusterer: 키워드 클러스터러
            enable_clustering: 클러스터링 활성화 여부
        """
        self.extractors = extractors or []
        self.similarity_analyzer = similarity_analyzer or KeywordSimilarityAnalyzer()
        self.clusterer = clusterer or KeywordClusterer() if enable_clustering else None
        self.enable_clustering = enable_clustering
        self.extraction_timeout = 180  # 각 추출기의 타임아웃 (초)
        logger.info(
            f"멀티 소스 키워드 매니저 초기화 "
            f"(추출기: {[e.name for e in self.extractors]}, 클러스터링: {enable_clustering})"
        )

    def add_extractor(self, extractor: BaseKeywordExtractor):
        """추출기 추가."""
        self.extractors.append(extractor)
        logger.info(f"{extractor.name} 추출기 추가됨")

    async def generate_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        max_keywords_per_source: int = 20,
        use_tiered_approach: bool = True,
        tier1_threshold: int = 15
    ) -> MultiSourceKeywordResult:
        """
        모든 소스에서 키워드를 생성하고 통합합니다.
        
        Tiered approach:
        - Tier 1: Fast/inexpensive extractors (Grok, Perplexity)
        - Tier 2: Advanced extractors (GPT-4o) if Tier 1 results are insufficient

        Args:
            topic: 키워드를 생성할 주제
            context: 추가 컨텍스트
            max_keywords_per_source: 소스당 최대 키워드 수
            use_tiered_approach: 계층적 접근 방식 사용 여부
            tier1_threshold: Tier 2를 트리거하는 최소 고유 키워드 수

        Returns:
            MultiSourceKeywordResult: 통합된 키워드 결과
        """
        start_time = time.time()
        logger.info(f"멀티 소스 키워드 생성 시작: '{topic}' (tiered={use_tiered_approach})")

        # 추출기를 계층별로 분류
        tier1_extractors = []  # Fast/inexpensive: Grok, Perplexity
        tier2_extractors = []  # Advanced: GPT-4o
        
        for extractor in self.extractors:
            if extractor.name.lower() in ['grok', 'perplexity']:
                tier1_extractors.append(extractor)
            else:
                tier2_extractors.append(extractor)
        
        source_results = {}
        keyword_lists = {}
        
        if use_tiered_approach and tier1_extractors:
            # Tier 1: 빠른/저렴한 추출기 먼저 실행
            logger.info(f"Tier 1 추출 시작: {[e.name for e in tier1_extractors]}")
            
            tier1_tasks = []
            for extractor in tier1_extractors:
                task = self._extract_with_timeout(
                    extractor,
                    topic,
                    context,
                    max_keywords_per_source
                )
                tier1_tasks.append(task)
            
            tier1_results = await asyncio.gather(*tier1_tasks, return_exceptions=True)
            
            # Tier 1 결과 처리
            for extractor, result in zip(tier1_extractors, tier1_results):
                if isinstance(result, Exception):
                    logger.error(f"{extractor.name} 추출 실패: {result}")
                    source_results[extractor.name] = KeywordExtractionResult(
                        keywords=[],
                        source_name=extractor.name,
                        extraction_time=0,
                        error=str(result)
                    )
                else:
                    source_results[extractor.name] = result
                    if result.is_success:
                        keyword_lists[extractor.name] = result.keywords
            
            # Tier 1 키워드 병합 및 평가
            tier1_merged = self._merge_keywords(keyword_lists)
            tier1_unique_high_confidence = [
                kw for kw in tier1_merged 
                if kw.confidence >= 0.7 and kw.importance in [KeywordImportance.HIGH, KeywordImportance.NORMAL]
            ]
            
            logger.info(f"Tier 1 결과: {len(tier1_unique_high_confidence)} 고신뢰 키워드")
            
            # Tier 2 필요 여부 결정
            if len(tier1_unique_high_confidence) < tier1_threshold and tier2_extractors:
                logger.info(f"Tier 2 추출 필요 (임계값 {tier1_threshold} 미달)")
                
                # Tier 1 키워드를 컨텍스트로 추가
                tier1_keywords_str = ", ".join([kw.keyword for kw in tier1_unique_high_confidence[:10]])
                enhanced_context = f"{context or ''}\nTier 1 keywords for refinement: {tier1_keywords_str}"
                
                # Tier 2 실행
                tier2_tasks = []
                for extractor in tier2_extractors:
                    task = self._extract_with_timeout(
                        extractor,
                        topic,
                        enhanced_context,
                        max_keywords_per_source
                    )
                    tier2_tasks.append(task)
                
                tier2_results = await asyncio.gather(*tier2_tasks, return_exceptions=True)
                
                # Tier 2 결과 처리
                for extractor, result in zip(tier2_extractors, tier2_results):
                    if isinstance(result, Exception):
                        logger.error(f"{extractor.name} 추출 실패: {result}")
                        source_results[extractor.name] = KeywordExtractionResult(
                            keywords=[],
                            source_name=extractor.name,
                            extraction_time=0,
                            error=str(result)
                        )
                    else:
                        source_results[extractor.name] = result
                        if result.is_success:
                            keyword_lists[extractor.name] = result.keywords
            else:
                logger.info("Tier 1 결과가 충분함, Tier 2 생략")
        else:
            # 기존 방식: 모든 소스 동시 실행
            extraction_tasks = []
            for extractor in self.extractors:
                task = self._extract_with_timeout(
                    extractor,
                    topic,
                    context,
                    max_keywords_per_source
                )
                extraction_tasks.append(task)
            
            extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            for extractor, result in zip(self.extractors, extraction_results):
                if isinstance(result, Exception):
                    logger.error(f"{extractor.name} 추출 실패: {result}")
                    source_results[extractor.name] = KeywordExtractionResult(
                        keywords=[],
                        source_name=extractor.name,
                        extraction_time=0,
                        error=str(result)
                    )
                else:
                    source_results[extractor.name] = result
                    if result.is_success:
                        keyword_lists[extractor.name] = result.keywords

        # 최종 키워드 병합 및 중복 제거
        merged_keywords = self._merge_keywords(keyword_lists)

        # 클러스터링 및 확장 키워드 생성
        clustering_result = None
        expansion_keywords = []
        
        if self.clusterer and self.enable_clustering and len(merged_keywords) >= 3:
            logger.info("키워드 클러스터링 및 확장 키워드 생성 시작")
            try:
                clustering_result = self.clusterer.cluster_and_expand(
                    keywords=merged_keywords,
                    topic=topic,
                    generate_expansion=True
                )
                expansion_keywords = clustering_result.expansion_keywords
                
                # 원본 키워드와 확장 키워드를 병합
                merged_keywords = self.clusterer.merge_with_original_keywords(
                    original_keywords=merged_keywords,
                    clustering_result=clustering_result
                )
                
                logger.success(f"클러스터링 완료: {len(expansion_keywords)}개 확장 키워드 생성")
                
            except Exception as e:
                logger.error(f"클러스터링 실패: {e}")
                clustering_result = None
                expansion_keywords = []

        # 중요도별 통계 계산 (확장 키워드 포함)
        stats = self._calculate_statistics(merged_keywords)

        # 최종 결과 생성
        result = MultiSourceKeywordResult(
            keywords=merged_keywords,
            source_results=source_results,
            total_time=time.time() - start_time,
            merged_count=stats['merged_count'],
            high_importance_count=stats['high_count'],
            normal_importance_count=stats['normal_count'],
            low_importance_count=stats['low_count'],
            expansion_keywords=expansion_keywords,
            clustering_result=clustering_result,
            metadata={
                'topic': topic,
                'context': context,
                'sources_used': list(keyword_lists.keys()),
                'total_sources': len(self.extractors),
                'tiered_approach_used': use_tiered_approach,
                'clustering_enabled': self.enable_clustering
            }
        )

        logger.success(
            f"멀티 소스 키워드 생성 완료: "
            f"총 {len(merged_keywords)}개 키워드 "
            f"(높음: {stats['high_count']}, 보통: {stats['normal_count']}, 낮음: {stats['low_count']})"
        )

        return result

    async def _extract_with_timeout(
        self,
        extractor: BaseKeywordExtractor,
        topic: str,
        context: Optional[str],
        max_keywords: int
    ) -> KeywordExtractionResult:
        """타임아웃이 있는 키워드 추출."""
        try:
            return await asyncio.wait_for(
                extractor.extract_keywords(topic, context, max_keywords),
                timeout=self.extraction_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"{extractor.name} 추출 타임아웃")
            return KeywordExtractionResult(
                keywords=[],
                source_name=extractor.name,
                extraction_time=self.extraction_timeout,
                error="Extraction timeout"
            )

    def _merge_keywords(self, keyword_lists: Dict[str, List[KeywordItem]]) -> List[KeywordItem]:
        """여러 소스의 키워드를 병합하고 중요도를 조정합니다."""
        # 유사도 분석기를 사용해 병합
        merged = self.similarity_analyzer.merge_similar_keywords(keyword_lists)

        # 중요도 기준으로 정렬 (HIGH > NORMAL > LOW)
        importance_order = {
            KeywordImportance.HIGH: 0,
            KeywordImportance.NORMAL: 1,
            KeywordImportance.LOW: 2
        }
        merged.sort(key=lambda x: (importance_order[x.importance], -x.confidence))

        return merged

    def _calculate_statistics(self, keywords: List[KeywordItem]) -> Dict[str, int]:
        """키워드 통계 계산."""
        high_count = sum(1 for kw in keywords if kw.importance == KeywordImportance.HIGH)
        normal_count = sum(1 for kw in keywords if kw.importance == KeywordImportance.NORMAL)
        low_count = sum(1 for kw in keywords if kw.importance == KeywordImportance.LOW)

        # 병합된 키워드 수 (2개 이상 소스에서 나온 키워드)
        merged_count = sum(1 for kw in keywords if len(kw.sources) >= 2)

        return {
            'high_count': high_count,
            'normal_count': normal_count,
            'low_count': low_count,
            'merged_count': merged_count
        }

    def get_top_keywords(
        self,
        result: MultiSourceKeywordResult,
        n: int = 10,
        importance_filter: Optional[List[KeywordImportance]] = None
    ) -> List[KeywordItem]:
        """
        상위 N개의 키워드를 반환합니다.

        Args:
            result: 멀티 소스 키워드 결과
            n: 반환할 키워드 수
            importance_filter: 필터링할 중요도 리스트

        Returns:
            List[KeywordItem]: 상위 키워드 리스트
        """
        keywords = result.keywords

        # 중요도 필터링
        if importance_filter:
            keywords = [kw for kw in keywords if kw.importance in importance_filter]

        return keywords[:n]