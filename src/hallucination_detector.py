"""
RePPL 기반 환각 탐지 및 응답 신뢰도 검증 모듈.

이 모듈은 "Repetition as Pre-Perplexity"(RePPL) 개념을 기반으로,
대규모 언어 모델(LLM)이 생성한 텍스트의 환각 현상을 탐지하고 신뢰도를 평가합니다.
주요 기능은 다음과 같습니다:
1.  **반복성 분석 (Repetition Analysis):** 텍스트 내 구문 반복을 측정합니다.
2.  **Perplexity 계산:** GPT-4o API의 logprobs를 활용하여 텍스트의 통계적 자연스러움을 평가합니다.
3.  **의미적 엔트로피 (Semantic Entropy):** 문장 간 의미적 다양성을 측정하여 내용의 풍부함을 평가합니다.
4.  **신뢰도 점수 종합:** 위의 세 가지 지표를 가중 합산하여 최종 신뢰도 점수를 도출합니다.
5.  **이슈 검색기 연동:** 생성된 이슈/콘텐츠를 RePPL로 검증하여 신뢰도 낮은 결과를 필터링합니다.
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
from sklearn.metrics.pairwise import cosine_similarity

# --- 내부 모듈 임포트 ---
# 프로젝트의 설정, 데이터 모델, 이슈 검색기, 키워드 생성기를 가져옵니다.
from src.config import config
from src.models import KeywordResult, IssueItem, SearchResult
from src.issue_searcher import create_issue_searcher
from src.keyword_generator import generate_keywords_for_topic


@dataclass
class RePPLScore:
    """RePPL 분석 결과를 저장하는 데이터 클래스.

    환각 탐지 모듈이 계산한 다양한 점수와 분석 세부 정보를 구조화하여 관리합니다.

    Attributes:
        repetition_score (float): 텍스트의 구문 반복 점수 (0~1). 높을수록 반복이 많음.
        perplexity (float): GPT-4o 기반 Perplexity 점수. 낮을수록 자연스러운 텍스트.
        semantic_entropy (float): 문장 간 의미적 다양성 점수 (0~1). 높을수록 다양한 주제를 다룸.
        confidence (float): 세 가지 지표를 종합한 최종 신뢰도 점수 (0~1). 높을수록 신뢰 가능.
        repeated_phrases (List[str]): 텍스트에서 2번 이상 반복된 구문 리스트.
        analysis_details (Dict[str, any]): 토큰 수, 문장 수 등 분석에 사용된 추가 메타데이터.
    """
    repetition_score: float
    perplexity: float
    semantic_entropy: float
    confidence: float
    repeated_phrases: List[str]
    analysis_details: Dict[str, any]


class RePPLHallucinationDetector:
    """RePPL 기반 환각 탐지기 (GPT-4o API 사용).

    텍스트의 반복성, Perplexity, 의미적 엔트로피를 분석하여
    LLM 응답의 신뢰도를 종합적으로 평가하는 클래스입니다.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        RePPLHallucinationDetector 인스턴스를 초기화합니다.

        Args:
            model_name (Optional[str]): Perplexity 계산에 사용할 OpenAI 모델 이름.
                                        지정하지 않으면 'gpt-4o'가 기본값으로 사용됩니다.
        """
        # 설정에서 OpenAI API 키를 가져옵니다.
        self.api_key = config.get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        # 비동기 OpenAI 클라이언트와 Perplexity 계산 모델을 설정합니다.
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.perplexity_model = model_name or "gpt-4o"

        # 문장 임베딩을 위한 Sentence Transformer 모델을 로드합니다.
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 신뢰도 계산에 사용될 임계값을 설정합니다.
        self.repetition_threshold = 0.3  # 반복 점수가 이 값을 넘으면 신뢰도 하락
        self.perplexity_threshold = 50   # Perplexity가 이 값을 넘으면 신뢰도 하락

        logger.info(f"RePPL 환각 탐지기 초기화 완료 (Perplexity 모델: {self.perplexity_model})")

    async def analyze_response(self, text: str, context: Optional[str] = None) -> RePPLScore:
        """주어진 텍스트에 대해 RePPL 분석을 수행하고 신뢰도 점수를 반환합니다.

        Args:
            text (str): 분석할 텍스트 (LLM 응답).
            context (Optional[str]): 텍스트의 주제나 맥락. Perplexity 계산의 정확도를 높이는 데 사용됩니다.

        Returns:
            RePPLScore: 반복성, Perplexity, 의미적 엔트로피, 최종 신뢰도 등이 포함된 분석 결과 객체.
        """
        logger.debug(f"RePPL 분석 시작 (텍스트 길이: {len(text)})")

        # 각 지표를 병렬 또는 순차적으로 계산합니다.
        repetition_score, repeated_phrases = self._analyze_repetition(text)
        perplexity = await self._calculate_perplexity_with_gpt(text, context)
        semantic_entropy = self._calculate_semantic_entropy(text)

        # 계산된 지표들을 종합하여 최종 신뢰도 점수를 도출합니다.
        confidence = self._calculate_confidence_score(
            repetition_score, perplexity, semantic_entropy
        )

        logger.debug(f"RePPL 점수 계산 완료: Repetition={repetition_score:.2f}, PPL={perplexity:.2f}, Entropy={semantic_entropy:.2f} -> Confidence={confidence:.2f}")

        # 분석에 사용된 추가 정보를 딕셔너리로 정리합니다.
        analysis_details = {
            "token_count": len(text.split()),
            "sentence_count": len(text.split('.')),
            "has_context": context is not None
        }

        # 최종 결과를 RePPLScore 객체로 만들어 반환합니다.
        return RePPLScore(
            repetition_score=repetition_score,
            perplexity=perplexity,
            semantic_entropy=semantic_entropy,
            confidence=confidence,
            repeated_phrases=repeated_phrases,
            analysis_details=analysis_details
        )

    def _analyze_repetition(self, text: str) -> Tuple[float, List[str]]:
        """텍스트 내 n-gram(연속된 단어 묶음)의 반복 패턴을 분석합니다.

        Args:
            text (str): 분석할 텍스트.

        Returns:
            Tuple[float, List[str]]:
                - 반복성 점수 (0.0 ~ 1.0).
                - 텍스트 내에서 2번 이상 나타난 구문들의 리스트.
        """
        phrases, repeated_phrases_set = [], set()

        # 💡 [수정] 구두점을 제거하고 소문자로 변환하여 단어 일치율을 높입니다.
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned_text.split()

        if not words: return 0.0, []

        # 3-gram부터 7-gram까지 반복되는 구문을 찾습니다.
        for n in range(3, 8):
            if len(words) < n: break
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            phrase_counts = Counter(ngrams)
            for phrase, count in phrase_counts.items():
                if count > 1:
                    repeated_phrases_set.add(' '.join(phrase))
                    phrases.append((' '.join(phrase), count))

        # 반복된 단어의 총 수를 계산하여 점수화합니다.
        repeated_words = sum(len(p.split()) * (c - 1) for p, c in phrases)
        repetition_score = min(1.0, repeated_words / len(words)) if len(words) > 0 else 0.0

        logger.debug(f"반복성 점수: {repetition_score:.3f}, 반복된 구문: {len(repeated_phrases_set)}개")
        return repetition_score, list(repeated_phrases_set)

    async def _calculate_perplexity_with_gpt(self, text: str, context: Optional[str] = None) -> float:
        """OpenAI GPT 모델을 사용하여 텍스트의 Perplexity를 계산합니다.

        Perplexity는 모델이 텍스트를 얼마나 '놀라워하는지'를 나타내며, 낮을수록 자연스럽고 예측 가능한 텍스트입니다.

        Args:
            text (str): Perplexity를 계산할 텍스트.
            context (Optional[str]): 텍스트의 맥락 정보.

        Returns:
            float: 계산된 Perplexity 값. 계산 실패 시 높은 임계값을 반환합니다.
        """
        if not text.strip(): return self.perplexity_threshold * 2  # 빈 텍스트는 최대 패널티
        logger.debug(f"Perplexity 계산 요청 (텍스트 길이: {len(text)})")
        try:
            # 컨텍스트가 있으면 프롬프트에 추가하여 정확도를 높입니다.
            full_text = f"Context: {context}\n\nText to evaluate: {text}" if context else text
            # OpenAI API를 호출하여 토큰별 로그 확률(logprobs)을 요청합니다.
            response = await self.client.chat.completions.create(
                model=self.perplexity_model,
                messages=[{"role": "user", "content": full_text}],
                max_tokens=2048,
                temperature=0,
                logprobs=True  # 💡 logprobs 활성화가 Perplexity 계산의 핵심
            )

            logprobs = [token.logprob for token in response.choices[0].logprobs.content if token.logprob is not None]
            if not logprobs: return self.perplexity_threshold * 2

            # 로그 확률의 평균을 구한 후, Perplexity 공식을 적용합니다. PPL = exp(-mean_log_prob)
            mean_logprob = sum(logprobs) / len(logprobs)
            perplexity = math.exp(-mean_logprob)
            logger.debug(f"GPT Perplexity 계산 성공: {perplexity:.2f}")
            return perplexity
        except Exception as e:
            # API 호출 실패나 계산 오류 발생 시 에러를 기록하고 패널티 값을 반환합니다.
            logger.error(f"GPT Perplexity 계산 실패: {e}")
            return self.perplexity_threshold * 2

    def _calculate_semantic_entropy(self, text: str) -> float:
        """문장 간 의미적 유사도를 기반으로 텍스트의 내용적 다양성(엔트로피)을 계산합니다.

        엔트로피가 높을수록 텍스트가 다양한 주제를 다루고 있음을 의미합니다.

        Args:
            text (str): 분석할 텍스트.

        Returns:
            float: 의미적 엔트로피 점수 (0.0 ~ 1.0). 0에 가까울수록 문장들이 의미적으로 유사함.
        """
        # 텍스트를 문장 단위로 분리하고, 너무 짧은 문장은 제외합니다.
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        if len(sentences) < 2:
            return 0.0  # 문장이 2개 미만이면 다양성 계산이 무의미

        # 각 문장을 임베딩 벡터로 변환합니다.
        embeddings = self.sentence_model.encode(sentences)
        # 문장 벡터 간의 코사인 유사도 행렬을 계산합니다.
        sim_matrix = cosine_similarity(embeddings)
        num_sentences = len(sentences)

        # 대각선(자기 자신과의 유사도)은 0으로 만들어 평균 계산에서 제외합니다.
        np.fill_diagonal(sim_matrix, 0)

        # 모든 문장 쌍의 평균 유사도를 계산합니다.
        avg_similarity = np.sum(sim_matrix) / (num_sentences * (num_sentences - 1))

        # 엔트로피를 '1 - 평균 유사도'로 정의합니다.
        entropy = 1 - avg_similarity
        logger.debug(f"의미적 엔트로피 계산됨: {entropy:.3f} (문장 {len(sentences)}개 기반)")
        return entropy

    def _calculate_confidence_score(self, repetition: float, perplexity: float, entropy: float) -> float:
        """세 가지 개별 점수를 가중 평균하여 최종 신뢰도 점수를 계산합니다.

        Args:
            repetition (float): 반복성 점수.
            perplexity (float): Perplexity 점수.
            entropy (float): 의미적 엔트로피 점수.

        Returns:
            float: 종합 신뢰도 점수 (0.0 ~ 1.0).
        """
        # 각 점수를 0~1 사이의 긍정적인 점수(높을수록 좋음)로 정규화합니다.
        rep_score = 1 - min(1.0, repetition / self.repetition_threshold)
        ppl_score = max(0.0, 1 - (perplexity / self.perplexity_threshold))
        ent_score = min(1.0, entropy)  # 엔트로피는 이미 0~1 범위이므로 그대로 사용

        # 각 지표에 대한 가중치를 정의합니다. (반복성에 가장 높은 가중치 부여)
        weights = {'repetition': 0.4, 'perplexity': 0.3, 'entropy': 0.3}

        # 가중 합산을 통해 최종 신뢰도 점수를 계산합니다.
        confidence = (
            weights['repetition'] * rep_score +
            weights['perplexity'] * ppl_score +
            weights['entropy'] * ent_score
        )
        return round(confidence, 3)


class RePPLEnhancedIssueSearcher:
    """RePPL 환각 탐지 모듈이 통합된 이슈 검색기.

    기본 이슈 검색기로 검색을 수행한 후, 각 결과물의 신뢰도를 RePPL로 검증하여
    품질이 낮은 콘텐츠를 필터링하는 역할을 합니다.
    """
    def __init__(self, api_key: Optional[str] = None):
        """RePPLEnhancedIssueSearcher 인스턴스를 초기화합니다.

        Args:
            api_key (Optional[str]): 이슈 검색에 사용할 API 키.
        """
        # 기본 이슈 검색기와 RePPL 탐지기를 초기화합니다.
        self.base_searcher = create_issue_searcher(api_key)
        self.reppl_detector = RePPLHallucinationDetector()

        # 검증을 통과하기 위한 최소 신뢰도 임계값을 설정합니다.
        self.min_confidence_threshold = 0.5
        logger.info("RePPL 강화 이슈 검색기 초기화 완료")

    async def search_with_validation(self, keyword_result: KeywordResult, time_period: str) -> SearchResult:
        """키워드로 이슈를 검색하고 RePPL로 각 결과를 검증합니다.

        신뢰도 높은 결과를 충분히 얻지 못하면, 키워드를 재생성하여 검색을 재시도합니다.

        Args:
            keyword_result (KeywordResult): 검색에 사용할 키워드 정보 객체.
            time_period (str): 검색할 기간 (예: 'past_week').

        Returns:
            SearchResult: RePPL 검증을 통과한 이슈 목록이 포함된 검색 결과 객체.
        """
        max_retries = 2
        current_keywords = keyword_result

        for attempt in range(max_retries):
            # 1. 기본 검색기로 이슈를 검색합니다.
            search_result = await self.base_searcher.search_issues_from_keywords(
                current_keywords, time_period, collect_details=True
            )
            validated_issues = []
            if search_result.issues:
                # 2. 검색된 각 이슈에 대해 RePPL 검증을 비동기적으로 수행합니다.
                validation_tasks = [self._validate_issue(issue, current_keywords.topic) for issue in search_result.issues]
                validation_results = await asyncio.gather(*validation_tasks)
                # 검증을 통과한 이슈(None이 아닌 결과)만 필터링합니다.
                validated_issues = [issue for issue in validation_results if issue is not None]

            # 3. 충분한 수의 신뢰도 높은 이슈를 찾았거나 마지막 시도이면 루프를 종료합니다.
            if len(validated_issues) >= 3 or attempt == max_retries - 1:
                search_result.issues = validated_issues
                search_result.total_found = len(validated_issues)
                return search_result

            # 4. 결과가 부족하면, 다른 관점의 키워드를 생성하여 재시도합니다.
            logger.info(f"신뢰도 높은 이슈 부족, 키워드를 재생성하여 재시도합니다. (시도 {attempt + 1}/{max_retries})")
            current_keywords = await generate_keywords_for_topic(f"{current_keywords.topic}의 다른 관점")

        return search_result

    async def _validate_issue(self, issue: IssueItem, topic: str) -> Optional[IssueItem]:
        """개별 이슈 항목의 콘텐츠를 RePPL로 분석하고 신뢰도를 평가합니다.

        Args:
            issue (IssueItem): 검증할 이슈 객체.
            topic (str): 이슈의 상위 주제 (분석 컨텍스트로 사용).

        Returns:
            Optional[IssueItem]: 신뢰도 임계값을 통과한 경우 해당 이슈 객체를, 그렇지 않으면 None을 반환합니다.
        """
        # 검증할 텍스트를 제목, 요약, 상세 내용 일부를 조합하여 구성합니다.
        content_to_validate = f"제목: {issue.title}\n요약: {issue.summary}"
        if issue.detailed_content:
            content_to_validate += f"\n상세내용: {issue.detailed_content[:500]}"

        # RePPL 분석을 수행합니다.
        reppl_score = await self.reppl_detector.analyze_response(content_to_validate, context=topic)

        # 계산된 신뢰도가 설정된 임계값 이상인 경우에만 이슈를 통과시킵니다.
        if reppl_score.confidence >= self.min_confidence_threshold:
            # 이슈 객체에 RePPL 신뢰도 및 분석 결과를 속성으로 추가합니다.
            setattr(issue, 'reppl_confidence', reppl_score.confidence)
            setattr(issue, 'reppl_analysis', reppl_score)
            logger.debug(f"이슈 '{issue.title[:30]}...' 검증 통과 (신뢰도: {reppl_score.confidence:.2f})")
            return issue
        else:
            # 임계값 미만인 경우 경고를 로그하고 해당 이슈를 제외(None 반환)합니다.
            logger.warning(f"이슈 '{issue.title[:30]}...' 제외됨 - RePPL 신뢰도: {reppl_score.confidence:.2f} < {self.min_confidence_threshold}")
            return None