"""
키워드 유사도 판별 및 중복 감지 모듈.
"""

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
from loguru import logger

from .base import KeywordItem

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