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

        self.api_key = config.get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.perplexity_model = model_name or "gpt-4o"
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.repetition_threshold = 0.3
        self.perplexity_threshold = 50
        self.is_initialized = True
        logger.info(f"RePPL 탐지기 초기화 완료 (Perplexity 모델: {self.perplexity_model})")

    async def analyze_text(self, text: str, context: Optional[str] = None) -> RePPLScore:
        """텍스트에 대해 RePPL 분석을 수행합니다."""
        logger.debug(f"RePPL 분석 시작 (텍스트 길이: {len(text)})")
        repetition_score, repeated_phrases = self._analyze_repetition(text)
        perplexity = await self._calculate_perplexity_with_gpt(text, context)
        semantic_entropy = self._calculate_semantic_entropy(text)
        confidence = self._calculate_confidence_score(repetition_score, perplexity, semantic_entropy)

        logger.debug(
            f"RePPL 분석 완료 - 반복성: {repetition_score:.2f}, "
            f"PPL: {perplexity:.2f}, 엔트로피: {semantic_entropy:.2f}, "
            f"신뢰도: {confidence:.2f}"
        )
        analysis_details = {
            "token_count": len(text.split()), "sentence_count": len(text.split('.')),
            "has_context": context is not None,
            "thresholds": {"repetition": self.repetition_threshold, "perplexity": self.perplexity_threshold}
        }
        return RePPLScore(
            confidence=confidence, repetition_score=repetition_score, perplexity=perplexity,
            semantic_entropy=semantic_entropy, repeated_phrases=repeated_phrases, analysis_details=analysis_details
        )

    def _analyze_repetition(self, text: str) -> Tuple[float, List[str]]:
        """
        텍스트 내 n-gram의 반복 패턴을 분석합니다. 중복 계산을 피하기 위해
        반복되는 단어의 인덱스를 추적하여 정확도를 높였습니다.
        """
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned_text.split()
        if len(words) < 3:
            return 0.0, []

        repeated_indices = set()
        all_repeated_phrases = set()

        for n in range(min(len(words), 7), 2, -1):
            ngrams = [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)

            for phrase, count in counts.items():
                if count > 1:
                    all_repeated_phrases.add(phrase)
                    # 이 구문이 나타나는 모든 위치의 인덱스를 기록
                    for i in range(len(words) - n + 1):
                        if ' '.join(words[i:i + n]) == phrase:
                            for j in range(n):
                                repeated_indices.add(i + j)

        repetition_score = len(repeated_indices) / len(words) if words else 0.0
        logger.debug(f"반복성 분석: 점수={repetition_score:.3f}, 반복 구문={len(all_repeated_phrases)}개")
        return repetition_score, sorted(list(all_repeated_phrases), key=len, reverse=True)

    async def _calculate_perplexity_with_gpt(self, text: str, context: Optional[str] = None) -> float:
        """OpenAI GPT를 사용하여 텍스트의 Perplexity를 계산합니다."""
        if not text.strip(): return self.perplexity_threshold * 2
        try:
            full_text = f"Context: {context}\n\nText: {text}" if context else text
            response = await self.client.chat.completions.create(
                model=self.perplexity_model, messages=[{"role": "user", "content": full_text}],
                max_tokens=2048, temperature=0, logprobs=True
            )
            logprobs = [t.logprob for t in response.choices[0].logprobs.content if t.logprob is not None]
            if not logprobs: return self.perplexity_threshold * 2
            return math.exp(-sum(logprobs) / len(logprobs))
        except Exception as e:
            logger.error(f"Perplexity 계산 실패: {e}")
            return self.perplexity_threshold * 2

    def _calculate_semantic_entropy(self, text: str) -> float:
        """문장 간 의미적 다양성(엔트로피)을 계산합니다."""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        if len(sentences) < 2: return 0.0
        embeddings = self.sentence_model.encode(sentences)
        sim_matrix = cosine_similarity(embeddings)
        if sim_matrix.shape[0] <= 1: return 0.0
        np.fill_diagonal(sim_matrix, 0)
        avg_similarity = np.sum(sim_matrix) / (len(sentences) * (len(sentences) - 1))
        return 1 - avg_similarity

    def _calculate_confidence_score(self, repetition: float, perplexity: float, entropy: float) -> float:
        """개별 점수를 종합하여 최종 신뢰도를 계산합니다."""
        rep_score = 1 - min(1.0, repetition / self.repetition_threshold)
        ppl_score = max(0.0, 1 - (perplexity / self.perplexity_threshold))
        ent_score = min(1.0, entropy)
        weights = {'repetition': 0.4, 'perplexity': 0.3, 'entropy': 0.3}
        confidence = (weights['repetition'] * rep_score + weights['perplexity'] * ppl_score + weights['entropy'] * ent_score)
        return round(confidence, 3)
