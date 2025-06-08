"""
RePPL (Repetition as Pre-Perplexity) 기반 환각 탐지기.

텍스트의 반복성, Perplexity, 의미적 엔트로피를 분석하여
LLM 응답의 신뢰도를 평가합니다.
"""

import re
import numpy as np
import math
from typing import List, Tuple, Optional
from collections import Counter
from loguru import logger
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config
from .base import BaseHallucinationDetector
from .models import RePPLScore


class RePPLDetector(BaseHallucinationDetector):
    """RePPL 기반 환각 탐지기 구현."""

    def __init__(self, model_name: Optional[str] = None):
        """
        RePPL 탐지기를 초기화합니다.

        Args:
            model_name (Optional[str]): Perplexity 계산에 사용할 OpenAI 모델
        """
        super().__init__("RePPL")

        # OpenAI 클라이언트 설정
        self.api_key = config.get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.perplexity_model = model_name or "gpt-4o"

        # 문장 임베딩 모델
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 임계값 설정
        self.repetition_threshold = 0.3
        self.perplexity_threshold = 50

        self.is_initialized = True
        logger.info(f"RePPL 탐지기 초기화 완료 (Perplexity 모델: {self.perplexity_model})")

    async def analyze_text(self, text: str, context: Optional[str] = None) -> RePPLScore:
        """
        텍스트에 대해 RePPL 분석을 수행합니다.

        Args:
            text (str): 분석할 텍스트
            context (Optional[str]): 텍스트의 맥락

        Returns:
            RePPLScore: 분석 결과
        """
        logger.debug(f"RePPL 분석 시작 (텍스트 길이: {len(text)})")

        # 각 지표 계산
        repetition_score, repeated_phrases = self._analyze_repetition(text)
        perplexity = await self._calculate_perplexity_with_gpt(text, context)
        semantic_entropy = self._calculate_semantic_entropy(text)

        # 최종 신뢰도 계산
        confidence = self._calculate_confidence_score(
            repetition_score, perplexity, semantic_entropy
        )

        logger.debug(
            f"RePPL 분석 완료 - 반복성: {repetition_score:.2f}, "
            f"PPL: {perplexity:.2f}, 엔트로피: {semantic_entropy:.2f}, "
            f"신뢰도: {confidence:.2f}"
        )

        # 분석 세부 정보
        analysis_details = {
            "token_count": len(text.split()),
            "sentence_count": len(text.split('.')),
            "has_context": context is not None,
            "thresholds": {
                "repetition": self.repetition_threshold,
                "perplexity": self.perplexity_threshold
            }
        }

        return RePPLScore(
            confidence=confidence,  # 부모 클래스 필드를 먼저
            repetition_score=repetition_score,
            perplexity=perplexity,
            semantic_entropy=semantic_entropy,
            repeated_phrases=repeated_phrases,
            analysis_details=analysis_details
        )

    def _analyze_repetition(self, text: str) -> Tuple[float, List[str]]:
        """
        텍스트 내 n-gram의 반복 패턴을 분석합니다.

        Args:
            text (str): 분석할 텍스트

        Returns:
            Tuple[float, List[str]]: 반복성 점수와 반복된 구문 리스트
        """
        phrases, repeated_phrases_set = [], set()

        # 텍스트 정제
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned_text.split()

        if not words:
            return 0.0, []

        # 3-gram부터 7-gram까지 분석
        for n in range(3, 8):
            if len(words) < n:
                break

            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            phrase_counts = Counter(ngrams)

            for phrase, count in phrase_counts.items():
                if count > 1:
                    phrase_str = ' '.join(phrase)
                    repeated_phrases_set.add(phrase_str)
                    phrases.append((phrase_str, count))

        # 반복된 단어 수 계산
        repeated_words = sum(len(p.split()) * (c - 1) for p, c in phrases)
        repetition_score = min(1.0, repeated_words / len(words)) if len(words) > 0 else 0.0

        logger.debug(f"반복성 분석: 점수={repetition_score:.3f}, 반복 구문={len(repeated_phrases_set)}개")

        return repetition_score, list(repeated_phrases_set)

    async def _calculate_perplexity_with_gpt(self, text: str, context: Optional[str] = None) -> float:
        """
        OpenAI GPT를 사용하여 텍스트의 Perplexity를 계산합니다.

        Args:
            text (str): 분석할 텍스트
            context (Optional[str]): 맥락 정보

        Returns:
            float: Perplexity 값
        """
        if not text.strip():
            return self.perplexity_threshold * 2

        logger.debug(f"Perplexity 계산 시작 (텍스트 길이: {len(text)})")

        try:
            # 컨텍스트 포함 텍스트 구성
            full_text = f"Context: {context}\n\nText: {text}" if context else text

            # API 호출
            response = await self.client.chat.completions.create(
                model=self.perplexity_model,
                messages=[{"role": "user", "content": full_text}],
                max_tokens=2048,
                temperature=0,
                logprobs=True
            )

            # 로그 확률 추출
            logprobs = [
                token.logprob
                for token in response.choices[0].logprobs.content
                if token.logprob is not None
            ]

            if not logprobs:
                return self.perplexity_threshold * 2

            # Perplexity 계산: PPL = exp(-mean_log_prob)
            mean_logprob = sum(logprobs) / len(logprobs)
            perplexity = math.exp(-mean_logprob)

            logger.debug(f"Perplexity 계산 완료: {perplexity:.2f}")
            return perplexity

        except Exception as e:
            logger.error(f"Perplexity 계산 실패: {e}")
            return self.perplexity_threshold * 2

    def _calculate_semantic_entropy(self, text: str) -> float:
        """
        문장 간 의미적 다양성(엔트로피)을 계산합니다.

        Args:
            text (str): 분석할 텍스트

        Returns:
            float: 의미적 엔트로피 (0.0 ~ 1.0)
        """
        # 문장 분리
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]

        if len(sentences) < 2:
            return 0.0

        # 문장 임베딩
        embeddings = self.sentence_model.encode(sentences)

        # 유사도 행렬 계산
        sim_matrix = cosine_similarity(embeddings)
        num_sentences = len(sentences)

        # 대각선 제외 (자기 자신과의 유사도)
        np.fill_diagonal(sim_matrix, 0)

        # 평균 유사도 계산
        avg_similarity = np.sum(sim_matrix) / (num_sentences * (num_sentences - 1))

        # 엔트로피 = 1 - 평균 유사도
        entropy = 1 - avg_similarity

        logger.debug(f"의미적 엔트로피: {entropy:.3f} (문장 {len(sentences)}개)")

        return entropy

    def _calculate_confidence_score(self, repetition: float, perplexity: float, entropy: float) -> float:
        """
        개별 점수를 종합하여 최종 신뢰도를 계산합니다.

        Args:
            repetition (float): 반복성 점수
            perplexity (float): Perplexity 점수
            entropy (float): 의미적 엔트로피

        Returns:
            float: 최종 신뢰도 (0.0 ~ 1.0)
        """
        # 각 점수 정규화
        rep_score = 1 - min(1.0, repetition / self.repetition_threshold)
        ppl_score = max(0.0, 1 - (perplexity / self.perplexity_threshold))
        ent_score = min(1.0, entropy)

        # 가중치
        weights = {
            'repetition': 0.4,
            'perplexity': 0.3,
            'entropy': 0.3
        }

        # 가중 평균
        confidence = (
            weights['repetition'] * rep_score +
            weights['perplexity'] * ppl_score +
            weights['entropy'] * ent_score
        )

        return round(confidence, 3)