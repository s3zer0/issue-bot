"""
RePPL 기반 환각 탐지 모듈 (GPT-4o API 기반 Perplexity 계산)
Repetition as Pre-Perplexity를 활용한 LLM 응답 신뢰도 검증
"""
import re
import numpy as np
import math
import asyncio
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
from loguru import logger
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
# 💡 [추가] scikit-learn에서 코사인 유사도 함수 import
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config
from src.models import KeywordResult, IssueItem, SearchResult
from src.issue_searcher import create_issue_searcher


@dataclass
class RePPLScore:
    """RePPL 분석 결과"""
    repetition_score: float
    perplexity: float
    semantic_entropy: float
    confidence: float
    repeated_phrases: List[str]
    analysis_details: Dict[str, any]


class RePPLHallucinationDetector:
    """RePPL 기반 환각 탐지기 (GPT-4o API 사용)"""

    def __init__(self, model_name: Optional[str] = None):
        self.api_key = config.get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.perplexity_model = model_name or "gpt-4o"

        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        self.repetition_threshold = 0.3
        self.perplexity_threshold = 50

        logger.info(f"RePPL 환각 탐지기 초기화 완료 (Perplexity 모델: {self.perplexity_model})")

    async def analyze_response(self, text: str, context: Optional[str] = None) -> RePPLScore:
        # ... (이전과 동일)
        logger.debug(f"RePPL 분석 시작 (텍스트 길이: {len(text)})")
        repetition_score, repeated_phrases = self._analyze_repetition(text)
        perplexity = await self._calculate_perplexity_with_gpt(text, context)
        semantic_entropy = self._calculate_semantic_entropy(text)
        confidence = self._calculate_confidence_score(repetition_score, perplexity, semantic_entropy)
        logger.debug(f"RePPL 점수 계산 완료: Repetition={repetition_score:.2f}, PPL={perplexity:.2f}, Entropy={semantic_entropy:.2f} -> Confidence={confidence:.2f}")
        analysis_details = {"token_count": len(text.split()),"sentence_count": len(text.split('.')),"has_context": context is not None}
        return RePPLScore(repetition_score=repetition_score, perplexity=perplexity, semantic_entropy=semantic_entropy, confidence=confidence, repeated_phrases=repeated_phrases, analysis_details=analysis_details)


    def _analyze_repetition(self, text: str) -> Tuple[float, List[str]]:
        # ... (이전과 동일)
        phrases, repeated_phrases_set = [], set()
        words = text.split()
        if not words: return 0.0, []
        for n in range(3, 8):
            if len(words) < n: break
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            phrase_counts = Counter(ngrams)
            for phrase, count in phrase_counts.items():
                if count > 1:
                    repeated_phrases_set.add(' '.join(phrase))
                    phrases.append((' '.join(phrase), count))
        repeated_words = sum(len(p.split()) * (c - 1) for p, c in phrases)
        repetition_score = min(1.0, repeated_words / len(words))
        logger.debug(f"반복성 점수: {repetition_score:.3f}, 반복된 구문: {len(repeated_phrases_set)}개")
        return repetition_score, list(repeated_phrases_set)


    async def _calculate_perplexity_with_gpt(self, text: str, context: Optional[str] = None) -> float:
        # ... (이전과 동일)
        if not text.strip(): return self.perplexity_threshold * 2
        logger.debug(f"Perplexity 계산 요청 (텍스트 길이: {len(text)})")
        try:
            full_text = f"Context: {context}\n\nText to evaluate: {text}" if context else text
            response = await self.client.chat.completions.create(model=self.perplexity_model, messages=[{"role": "user", "content": full_text}], max_tokens=2048, temperature=0, logprobs=True)
            logprobs = [token.logprob for token in response.choices[0].logprobs.content if token.logprob is not None]
            if not logprobs: return self.perplexity_threshold * 2
            mean_logprob = sum(logprobs) / len(logprobs)
            perplexity = math.exp(-mean_logprob)
            logger.debug(f"GPT Perplexity 계산 성공: {perplexity:.2f}")
            return perplexity
        except Exception as e:
            logger.error(f"GPT Perplexity 계산 실패: {e}")
            return self.perplexity_threshold * 2

    # 💡 [핵심 수정] np.inner 대신 sklearn.metrics.pairwise.cosine_similarity 사용
    def _calculate_semantic_entropy(self, text: str) -> float:
        """문장 간 의미적 다양성(엔트로피) 계산 (안정성 강화)"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        if len(sentences) < 2:
            return 0.0

        embeddings = self.sentence_model.encode(sentences)

        # scikit-learn을 사용하여 코사인 유사도 행렬 계산
        sim_matrix = cosine_similarity(embeddings)

        # 자기 자신과의 유사도(대각선)를 제외한 모든 유사도의 평균 계산
        num_sentences = len(sentences)
        # 대각선을 0으로 만들어 합산에서 제외
        np.fill_diagonal(sim_matrix, 0)
        # off-diagonal 요소들의 합을 개수(N*(N-1))로 나눔
        avg_similarity = np.sum(sim_matrix) / (num_sentences * (num_sentences - 1))

        entropy = 1 - avg_similarity
        logger.debug(f"의미적 엔트로피 계산됨: {entropy:.3f} (문장 {len(sentences)}개 기반)")
        return entropy

    def _calculate_confidence_score(self, repetition: float, perplexity: float, entropy: float) -> float:
        """통합 신뢰도 점수 계산"""
        # (이전과 동일)
        rep_score = 1 - min(1.0, repetition / self.repetition_threshold)
        ppl_score = max(0.0, 1 - (perplexity / self.perplexity_threshold))
        ent_score = min(1.0, entropy)
        weights = {'repetition': 0.4, 'perplexity': 0.3, 'entropy': 0.3}
        confidence = (weights['repetition'] * rep_score + weights['perplexity'] * ppl_score + weights['entropy'] * ent_score)
        return round(confidence, 3)


class RePPLEnhancedIssueSearcher:
    # ... (이하 클래스 코드는 이전과 동일)
    """RePPL이 통합된 이슈 검색기"""
    def __init__(self, api_key: Optional[str] = None):
        self.base_searcher = create_issue_searcher(api_key)
        self.reppl_detector = RePPLHallucinationDetector()
        self.min_confidence_threshold = 0.5 # 신뢰도 임계값
        logger.info("RePPL 강화 이슈 검색기 초기화 완료")

    async def search_with_validation(self, keyword_result: KeywordResult, time_period: str) -> SearchResult:
        """RePPL 검증이 포함된 이슈 검색"""
        max_retries = 2
        current_keywords = keyword_result

        for attempt in range(max_retries):
            search_result = await self.base_searcher.search_issues_from_keywords(current_keywords, time_period, collect_details=True)
            validated_issues = []
            if search_result.issues:
                validation_tasks = [self._validate_issue(issue, current_keywords.topic) for issue in search_result.issues]
                validation_results = await asyncio.gather(*validation_tasks)
                validated_issues = [issue for issue in validation_results if issue is not None]

            if len(validated_issues) >= 3 or attempt == max_retries - 1:
                search_result.issues = validated_issues
                search_result.total_found = len(validated_issues)
                return search_result

            logger.info(f"신뢰도 높은 이슈 부족, 키워드를 재생성하여 재시도합니다.")
            from src.keyword_generator import generate_keywords_for_topic
            current_keywords = await generate_keywords_for_topic(f"{current_keywords.topic}의 다른 관점")
        return search_result

    async def _validate_issue(self, issue: IssueItem, topic: str) -> Optional[IssueItem]:
        """개별 이슈를 RePPL로 검증하고 신뢰도 정보를 추가합니다."""
        content_to_validate = f"제목: {issue.title}\n요약: {issue.summary}"
        if issue.detailed_content:
            content_to_validate += f"\n상세내용: {issue.detailed_content[:500]}"
        reppl_score = await self.reppl_detector.analyze_response(content_to_validate, context=topic)
        if reppl_score.confidence >= self.min_confidence_threshold:
            setattr(issue, 'reppl_confidence', reppl_score.confidence)
            setattr(issue, 'reppl_analysis', reppl_score)
            logger.debug(f"이슈 '{issue.title[:30]}...' 검증 통과 (신뢰도: {reppl_score.confidence:.2f})")
            return issue
        else:
            logger.warning(f"이슈 '{issue.title[:30]}...' 제외됨 - RePPL 신뢰도: {reppl_score.confidence:.2f}")
            return None