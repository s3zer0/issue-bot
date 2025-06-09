# src/keyword_generation/base.py
"""
키워드 생성 시스템의 기본 인터페이스 및 데이터 모델 정의.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum
import asyncio
from loguru import logger


class KeywordImportance(Enum):
    """키워드 중요도 레벨."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class KeywordItem:
    """개별 키워드와 메타데이터."""
    keyword: str
    sources: List[str] = field(default_factory=list)
    importance: KeywordImportance = KeywordImportance.NORMAL
    confidence: float = 0.5
    category: Optional[str] = None  # primary, related, context
    metadata: Dict[str, any] = field(default_factory=dict)

    def add_source(self, source: str):
        """소스 추가 및 중요도 자동 업데이트."""
        if source not in self.sources:
            self.sources.append(source)
            # 2개 이상 소스에서 나온 키워드는 HIGH
            if len(self.sources) >= 2:
                self.importance = KeywordImportance.HIGH

    def merge_with(self, other: 'KeywordItem'):
        """다른 키워드 아이템과 병합."""
        for source in other.sources:
            self.add_source(source)
        # 더 높은 신뢰도 채택
        self.confidence = max(self.confidence, other.confidence)
        # 메타데이터 병합
        self.metadata.update(other.metadata)


@dataclass
class KeywordExtractionResult:
    """키워드 추출 결과."""
    keywords: List[KeywordItem]
    source_name: str
    extraction_time: float
    raw_response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """추출 성공 여부."""
        return self.error is None and len(self.keywords) > 0


class BaseKeywordExtractor(ABC):
    """키워드 추출기의 기본 추상 클래스."""

    def __init__(self, name: str, api_key: Optional[str] = None):
        """
        키워드 추출기 초기화.

        Args:
            name: 추출기 이름 (e.g., "GPT", "Grok", "Perplexity")
            api_key: API 키 (필요한 경우)
        """
        self.name = name
        self.api_key = api_key
        self.is_initialized = False
        logger.info(f"{self.name} 키워드 추출기 초기화")

    @abstractmethod
    async def extract_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        max_keywords: int = 20
    ) -> KeywordExtractionResult:
        """
        주제로부터 키워드를 추출합니다.

        Args:
            topic: 키워드를 추출할 주제
            context: 추가 컨텍스트 정보
            max_keywords: 최대 키워드 수

        Returns:
            KeywordExtractionResult: 추출 결과
        """
        pass

    def preprocess_topic(self, topic: str) -> str:
        """주제 전처리 (공통 로직)."""
        return topic.strip().lower()

    def postprocess_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 후처리 (중복 제거, 정규화 등)."""
        # 소문자 변환 및 공백 제거
        processed = [kw.strip().lower() for kw in keywords]
        # 중복 제거 (순서 유지)
        seen = set()
        unique = []
        for kw in processed:
            if kw and kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique


# src/keyword_generation/similarity.py
"""
키워드 유사도 판별 및 중복 감지 모듈.
"""

from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
from loguru import logger


@dataclass
class SimilarityResult:
    """유사도 분석 결과."""
    is_similar: bool
    similarity_score: float
    similarity_type: str  # exact, semantic, fuzzy
    matched_keyword: Optional[str] = None


class KeywordSimilarityAnalyzer:
    """키워드 유사도 분석기."""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        유사도 분석기 초기화.

        Args:
            similarity_threshold: 유사하다고 판단할 임계값 (0.0 ~ 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.sentence_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self._init_synonym_dict()
        logger.info(f"키워드 유사도 분석기 초기화 (임계값: {similarity_threshold})")

    def _init_synonym_dict(self):
        """동의어 사전 초기화."""
        self.synonym_dict = {
            # 영어 동의어
            'ai': ['artificial intelligence', 'machine intelligence'],
            'ml': ['machine learning'],
            'dl': ['deep learning'],
            'nlp': ['natural language processing'],
            'cv': ['computer vision'],
            'iot': ['internet of things'],
            'api': ['application programming interface'],
            # 한글 동의어
            '인공지능': ['ai', 'artificial intelligence'],
            '머신러닝': ['ml', 'machine learning', '기계학습'],
            '딥러닝': ['dl', 'deep learning', '심층학습'],
            '자연어처리': ['nlp', '자연어 처리'],
            '컴퓨터비전': ['cv', '컴퓨터 비전'],
            '사물인터넷': ['iot'],
            # 기술 용어 변형
            'gpt': ['gpt-3', 'gpt-4', 'gpt3', 'gpt4'],
            'bert': ['bert-base', 'bert-large'],
        }

    def find_similar_keywords(
        self,
        keyword: str,
        keyword_pool: List[str]
    ) -> List[Tuple[str, SimilarityResult]]:
        """
        키워드 풀에서 유사한 키워드를 찾습니다.

        Args:
            keyword: 찾을 키워드
            keyword_pool: 비교할 키워드 리스트

        Returns:
            List[Tuple[str, SimilarityResult]]: (유사 키워드, 유사도 결과) 튜플 리스트
        """
        similar_keywords = []

        for candidate in keyword_pool:
            # 1. 완전 일치 검사
            if self._exact_match(keyword, candidate):
                result = SimilarityResult(
                    is_similar=True,
                    similarity_score=1.0,
                    similarity_type='exact',
                    matched_keyword=candidate
                )
                similar_keywords.append((candidate, result))
                continue

            # 2. 동의어 검사
            if self._synonym_match(keyword, candidate):
                result = SimilarityResult(
                    is_similar=True,
                    similarity_score=0.95,
                    similarity_type='synonym',
                    matched_keyword=candidate
                )
                similar_keywords.append((candidate, result))
                continue

            # 3. 문자열 유사도 검사 (Fuzzy Matching)
            fuzzy_score = self._fuzzy_match(keyword, candidate)
            if fuzzy_score >= self.similarity_threshold:
                result = SimilarityResult(
                    is_similar=True,
                    similarity_score=fuzzy_score,
                    similarity_type='fuzzy',
                    matched_keyword=candidate
                )
                similar_keywords.append((candidate, result))
                continue

            # 4. 의미적 유사도 검사 (Semantic Similarity)
            semantic_score = self._semantic_match(keyword, candidate)
            if semantic_score >= self.similarity_threshold:
                result = SimilarityResult(
                    is_similar=True,
                    similarity_score=semantic_score,
                    similarity_type='semantic',
                    matched_keyword=candidate
                )
                similar_keywords.append((candidate, result))

        return similar_keywords

    def _exact_match(self, kw1: str, kw2: str) -> bool:
        """완전 일치 검사."""
        return kw1.lower().strip() == kw2.lower().strip()

    def _synonym_match(self, kw1: str, kw2: str) -> bool:
        """동의어 매칭 검사."""
        kw1_lower = kw1.lower().strip()
        kw2_lower = kw2.lower().strip()

        # 직접 동의어 관계 확인
        if kw1_lower in self.synonym_dict:
            if kw2_lower in self.synonym_dict[kw1_lower]:
                return True

        if kw2_lower in self.synonym_dict:
            if kw1_lower in self.synonym_dict[kw2_lower]:
                return True

        return False

    def _fuzzy_match(self, kw1: str, kw2: str) -> float:
        """문자열 유사도 계산 (레벤슈타인 거리 기반)."""
        # 전처리: 공백, 특수문자 정규화
        kw1_normalized = re.sub(r'[^a-zA-Z0-9가-힣]', '', kw1.lower())
        kw2_normalized = re.sub(r'[^a-zA-Z0-9가-힣]', '', kw2.lower())

        # SequenceMatcher를 사용한 유사도 계산
        return SequenceMatcher(None, kw1_normalized, kw2_normalized).ratio()

    def _semantic_match(self, kw1: str, kw2: str) -> float:
        """의미적 유사도 계산 (문장 임베딩 기반)."""
        try:
            # 문장 임베딩 생성
            embeddings = self.sentence_model.encode([kw1, kw2])
            # 코사인 유사도 계산
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            return 0.0

    def merge_similar_keywords(
        self,
        keyword_lists: Dict[str, List[KeywordItem]]
    ) -> List[KeywordItem]:
        """
        여러 소스의 키워드를 병합하고 중복을 제거합니다.

        Args:
            keyword_lists: 소스별 키워드 리스트 딕셔너리

        Returns:
            List[KeywordItem]: 병합된 키워드 리스트
        """
        merged_keywords = []
        processed_keywords = set()

        # 모든 키워드를 순회하며 병합
        for source, keywords in keyword_lists.items():
            for kw_item in keywords:
                # 이미 처리된 유사 키워드가 있는지 확인
                similar_found = False

                for merged_kw in merged_keywords:
                    similar_results = self.find_similar_keywords(
                        kw_item.keyword,
                        [merged_kw.keyword]
                    )

                    if similar_results:
                        # 유사한 키워드가 있으면 병합
                        merged_kw.merge_with(kw_item)
                        similar_found = True
                        break

                if not similar_found:
                    # 새로운 키워드 추가
                    new_item = KeywordItem(
                        keyword=kw_item.keyword,
                        sources=[source],
                        confidence=kw_item.confidence,
                        category=kw_item.category,
                        metadata=kw_item.metadata.copy()
                    )
                    merged_keywords.append(new_item)

        return merged_keywords


# src/keyword_generation/manager.py
"""
멀티 소스 키워드 생성 매니저.
"""

from typing import List, Dict, Optional, Set
import asyncio
import time
from dataclasses import dataclass, field
from loguru import logger

from .base import BaseKeywordExtractor, KeywordExtractionResult, KeywordItem, KeywordImportance
from .similarity import KeywordSimilarityAnalyzer
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
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_legacy_format(self, topic: str) -> KeywordResult:
        """기존 KeywordResult 형식으로 변환 (호환성)."""
        # 카테고리별로 키워드 분류
        primary = []
        related = []
        context = []

        for kw_item in self.keywords:
            if kw_item.category == 'primary' or kw_item.importance == KeywordImportance.HIGH:
                primary.append(kw_item.keyword)
            elif kw_item.category == 'related':
                related.append(kw_item.keyword)
            else:
                context.append(kw_item.keyword)

        # 전체 신뢰도 계산 (평균)
        avg_confidence = sum(kw.confidence for kw in self.keywords) / len(self.keywords) if self.keywords else 0.5

        return KeywordResult(
            topic=topic,
            primary_keywords=primary[:10],  # 최대 10개
            related_terms=related[:10],
            context_keywords=context[:10],
            confidence_score=avg_confidence,
            generation_time=self.total_time,
            raw_response=str(self.metadata)
        )


class MultiSourceKeywordManager:
    """여러 소스를 통합하여 키워드를 생성하는 매니저."""

    def __init__(
        self,
        extractors: Optional[List[BaseKeywordExtractor]] = None,
        similarity_analyzer: Optional[KeywordSimilarityAnalyzer] = None
    ):
        """
        멀티 소스 키워드 매니저 초기화.

        Args:
            extractors: 사용할 키워드 추출기 리스트
            similarity_analyzer: 유사도 분석기
        """
        self.extractors = extractors or []
        self.similarity_analyzer = similarity_analyzer or KeywordSimilarityAnalyzer()
        self.extraction_timeout = 30  # 각 추출기의 타임아웃 (초)
        logger.info(
            f"멀티 소스 키워드 매니저 초기화 "
            f"(추출기: {[e.name for e in self.extractors]})"
        )

    def add_extractor(self, extractor: BaseKeywordExtractor):
        """추출기 추가."""
        self.extractors.append(extractor)
        logger.info(f"{extractor.name} 추출기 추가됨")

    async def generate_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        max_keywords_per_source: int = 20
    ) -> MultiSourceKeywordResult:
        """
        모든 소스에서 키워드를 생성하고 통합합니다.

        Args:
            topic: 키워드를 생성할 주제
            context: 추가 컨텍스트
            max_keywords_per_source: 소스당 최대 키워드 수

        Returns:
            MultiSourceKeywordResult: 통합된 키워드 결과
        """
        start_time = time.time()
        logger.info(f"멀티 소스 키워드 생성 시작: '{topic}'")

        # 1. 모든 소스에서 병렬로 키워드 추출
        extraction_tasks = []
        for extractor in self.extractors:
            task = self._extract_with_timeout(
                extractor,
                topic,
                context,
                max_keywords_per_source
            )
            extraction_tasks.append(task)

        # 모든 태스크 실행 및 결과 수집
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # 2. 결과 정리 및 에러 처리
        source_results = {}
        keyword_lists = {}

        for extractor, result in zip(self.extractors, extraction_results):
            if isinstance(result, Exception):
                logger.error(f"{extractor.name} 추출 실패: {result}")
                # 실패한 경우도 기록
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

        # 3. 키워드 병합 및 중복 제거
        merged_keywords = self._merge_keywords(keyword_lists)

        # 4. 중요도별 통계 계산
        stats = self._calculate_statistics(merged_keywords)

        # 5. 최종 결과 생성
        result = MultiSourceKeywordResult(
            keywords=merged_keywords,
            source_results=source_results,
            total_time=time.time() - start_time,
            merged_count=stats['merged_count'],
            high_importance_count=stats['high_count'],
            normal_importance_count=stats['normal_count'],
            low_importance_count=stats['low_count'],
            metadata={
                'topic': topic,
                'context': context,
                'sources_used': list(keyword_lists.keys()),
                'total_sources': len(self.extractors)
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