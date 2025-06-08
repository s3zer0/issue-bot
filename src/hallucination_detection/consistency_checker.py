"""
자기 일관성 검사기 (Self-Consistency Checker).

동일한 질문에 대해 여러 번의 LLM 응답을 생성하고,
응답 간의 일관성을 분석하여 환각 현상을 탐지합니다.
"""

import asyncio
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from difflib import SequenceMatcher
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config
from src.clients.perplexity_client import PerplexityClient
from .base import BaseHallucinationDetector
from .models import ConsistencyScore


class SelfConsistencyChecker(BaseHallucinationDetector):
    """자기 일관성 검사를 통한 환각 탐지기."""

    def __init__(self, num_queries: int = 3, similarity_threshold: float = 0.7):
        """
        자기 일관성 검사기를 초기화합니다.

        Args:
            num_queries (int): 일관성 검사를 위한 쿼리 횟수
            similarity_threshold (float): 일관성 판단 임계값
        """
        super().__init__("Self-Consistency")

        self.num_queries = num_queries
        self.similarity_threshold = similarity_threshold

        # Perplexity 클라이언트
        self.perplexity_client = PerplexityClient()

        # 문장 임베딩 모델
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        self.is_initialized = True
        logger.info(f"자기 일관성 검사기 초기화 완료 (쿼리 수: {self.num_queries})")

    async def analyze_text(self, text: str, context: Optional[str] = None) -> ConsistencyScore:
        """
        텍스트에 대한 자기 일관성을 검사합니다.

        Args:
            text (str): 분석할 텍스트 (원본 응답)
            context (Optional[str]): 원래 질문이나 주제

        Returns:
            ConsistencyScore: 일관성 분석 결과
        """
        if not context:
            # 텍스트에서 주제 추출 시도
            context = self._extract_topic_from_text(text)

        logger.info(f"자기 일관성 검사 시작 (주제: {context[:50]}...)")

        # 여러 번의 응답 생성
        responses = await self._generate_multiple_responses(context)

        # 원본 텍스트도 응답 목록에 추가
        all_responses = [text] + responses

        # 일관성 분석
        consistency_rate, variations = self._analyze_consistency(all_responses)

        # 공통 요소 및 차이점 추출
        common_elements = self._extract_common_elements(all_responses)
        divergent_elements = self._extract_divergent_elements(all_responses)

        # 일관된 응답 수 계산
        num_consistent = self._count_consistent_responses(all_responses)

        # 최종 신뢰도 계산
        confidence = self._calculate_confidence_from_consistency(
            consistency_rate,
            num_consistent,
            len(all_responses)
        )

        logger.info(
            f"자기 일관성 검사 완료 - 일치율: {consistency_rate:.2f}, "
            f"신뢰도: {confidence:.2f}"
        )

        return ConsistencyScore(
            confidence=confidence,  # 부모 클래스 필드를 먼저
            consistency_rate=consistency_rate,
            num_queries=len(all_responses),
            num_consistent=num_consistent,
            variations=variations[:5],  # 상위 5개 변형만 저장
            common_elements=common_elements[:10],  # 상위 10개 공통 요소
            divergent_elements=divergent_elements[:10],  # 상위 10개 차이점
            analysis_details={
                "original_text_included": True,
                "similarity_threshold": self.similarity_threshold,
                "average_similarity": self._calculate_average_similarity(all_responses)
            }
        )

    async def _generate_multiple_responses(self, query: str) -> List[str]:
        """
        동일한 질문에 대해 여러 개의 응답을 생성합니다.

        Args:
            query (str): 질문 또는 주제

        Returns:
            List[str]: 생성된 응답 리스트
        """
        responses = []

        # 병렬로 여러 응답 생성
        tasks = []
        for i in range(self.num_queries - 1):  # 원본 텍스트를 제외한 수
            # 약간씩 다른 프롬프트로 다양성 확보
            prompt_variation = self._create_prompt_variation(query, i)
            task = self._get_single_response(prompt_variation)
            tasks.append(task)

        # 모든 응답 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"응답 생성 실패 ({i+1}): {result}")
            else:
                responses.append(result)

        return responses

    async def _get_single_response(self, prompt: str) -> str:
        """
        단일 응답을 생성합니다.

        Args:
            prompt (str): 프롬프트

        Returns:
            str: 생성된 응답
        """
        try:
            response = await self.perplexity_client._make_api_call(prompt)
            content = response['choices'][0]['message']['content']
            return content
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            raise

    def _create_prompt_variation(self, query: str, variation_idx: int) -> str:
        """
        약간의 변형을 가진 프롬프트를 생성합니다.

        Args:
            query (str): 원본 질의
            variation_idx (int): 변형 인덱스

        Returns:
            str: 변형된 프롬프트
        """
        variations = [
            f"{query}에 대해 자세히 설명해주세요.",
            f"{query}와 관련된 최신 정보를 알려주세요.",
            f"{query}에 대한 핵심 내용을 요약해주세요.",
            f"{query}의 중요한 측면들을 설명해주세요.",
            f"{query}에 대해 알아야 할 주요 사항은 무엇인가요?"
        ]

        return variations[variation_idx % len(variations)]

    def _analyze_consistency(self, responses: List[str]) -> Tuple[float, List[str]]:
        """
        응답들 간의 일관성을 분석합니다.

        Args:
            responses (List[str]): 분석할 응답 리스트

        Returns:
            Tuple[float, List[str]]: 일관성 비율과 주요 변형들
        """
        if len(responses) < 2:
            return 1.0, []

        # 의미적 유사도 기반 일관성 계산
        embeddings = self.sentence_model.encode(responses)
        similarity_matrix = cosine_similarity(embeddings)

        # 응답별 핵심 요소 추출
        key_elements_per_response = [self._extract_key_elements(resp) for resp in responses]

        # 요소 간 일치도와 의미적 유사도를 모두 고려
        total_comparisons = 0
        consistent_comparisons = 0
        variations = []

        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # 의미적 유사도
                semantic_sim = similarity_matrix[i][j]

                # 요소 기반 유사도
                element_sim = self._calculate_element_similarity(
                    key_elements_per_response[i],
                    key_elements_per_response[j]
                )

                # 통합 유사도 (의미적 유사도와 요소 유사도의 평균)
                combined_similarity = (semantic_sim + element_sim) / 2

                total_comparisons += 1
                if combined_similarity >= self.similarity_threshold:
                    consistent_comparisons += 1
                else:
                    # 차이점 기록
                    diff = self._find_differences(responses[i], responses[j])
                    if diff:
                        variations.append(diff)

        consistency_rate = consistent_comparisons / total_comparisons if total_comparisons > 0 else 0.0

        # 중복 제거 및 빈도순 정렬
        variation_counter = Counter(variations)
        unique_variations = [var for var, _ in variation_counter.most_common()]

        return consistency_rate, unique_variations

    def _extract_key_elements(self, text: str) -> Set[str]:
        """
        텍스트에서 핵심 요소를 추출합니다.

        Args:
            text (str): 분석할 텍스트

        Returns:
            Set[str]: 핵심 요소 집합
        """
        # 숫자, 날짜, 고유명사, 주요 명사구 추출
        elements = set()

        # 숫자 추출 (퍼센트 포함)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        elements.update(numbers)

        # 한글 날짜 패턴 (2024년, 2024년 3월, 2024년 3월 15일)
        korean_dates = re.findall(r'\b\d{4}년(?:\s*\d{1,2}월)?(?:\s*\d{1,2}일)?\b', text)
        elements.update(korean_dates)

        # 영어 날짜 패턴
        english_dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', text)
        elements.update(english_dates)

        # 단순 연도
        years = re.findall(r'\b20\d{2}년?\b', text)
        elements.update(years)

        # 인용구
        quotes = re.findall(r'"([^"]+)"', text)
        elements.update(quotes)

        # 영어 대문자로 시작하는 단어들 (고유명사)
        english_proper = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        elements.update(english_proper)

        # 영어 약어 (모두 대문자)
        acronyms = re.findall(r'\b[A-Z]{2,}(?:-\d+)?\b', text)  # GPT-5 같은 패턴도 포함
        elements.update(acronyms)

        return elements

    def _calculate_element_similarity(self, elements1: Set[str], elements2: Set[str]) -> float:
        """
        두 요소 집합 간의 유사도를 계산합니다.

        Args:
            elements1 (Set[str]): 첫 번째 요소 집합
            elements2 (Set[str]): 두 번째 요소 집합

        Returns:
            float: 유사도 (0.0 ~ 1.0)
        """
        if not elements1 and not elements2:
            return 1.0
        if not elements1 or not elements2:
            return 0.0

        intersection = elements1.intersection(elements2)
        union = elements1.union(elements2)

        return len(intersection) / len(union)

    def _find_differences(self, text1: str, text2: str) -> Optional[str]:
        """
        두 텍스트 간의 주요 차이점을 찾습니다.

        Args:
            text1 (str): 첫 번째 텍스트
            text2 (str): 두 번째 텍스트

        Returns:
            Optional[str]: 주요 차이점 설명
        """
        # 문장 단위로 분할
        sentences1 = [s.strip() for s in text1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in text2.split('.') if s.strip()]

        # 유사도가 낮은 문장 쌍 찾기
        matcher = SequenceMatcher(None, sentences1, sentences2)
        differences = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                if i1 < len(sentences1) and j1 < len(sentences2):
                    diff = f"불일치: '{sentences1[i1][:50]}...' vs '{sentences2[j1][:50]}...'"
                    differences.append(diff)
            elif tag == 'delete':
                if i1 < len(sentences1):
                    diff = f"첫 번째에만 존재: '{sentences1[i1][:50]}...'"
                    differences.append(diff)
            elif tag == 'insert':
                if j1 < len(sentences2):
                    diff = f"두 번째에만 존재: '{sentences2[j1][:50]}...'"
                    differences.append(diff)

        return differences[0] if differences else None

    def _extract_common_elements(self, responses: List[str]) -> List[str]:
        """
        모든 응답에서 공통적으로 나타나는 요소를 추출합니다.

        Args:
            responses (List[str]): 응답 리스트

        Returns:
            List[str]: 공통 요소 리스트
        """
        if not responses:
            return []

        # 각 응답에서 요소 추출
        all_elements = [self._extract_key_elements(resp) for resp in responses]

        # 모든 응답에 공통으로 나타나는 요소
        common = set.intersection(*all_elements) if all_elements else set()

        return list(common)

    def _extract_divergent_elements(self, responses: List[str]) -> List[str]:
        """
        응답 간 차이를 보이는 요소를 추출합니다.

        Args:
            responses (List[str]): 응답 리스트

        Returns:
            List[str]: 차이점 리스트
        """
        if len(responses) < 2:
            return []

        divergent = []
        all_elements = [self._extract_key_elements(resp) for resp in responses]

        # 일부 응답에만 나타나는 요소 찾기
        element_counts = Counter()
        for elements in all_elements:
            element_counts.update(elements)

        # 전체 응답의 절반 미만에서만 나타나는 요소
        threshold = len(responses) / 2
        for element, count in element_counts.items():
            if count < threshold:
                divergent.append(f"{element} (등장: {count}/{len(responses)})")

        return divergent[:10]  # 상위 10개만

    def _count_consistent_responses(self, responses: List[str]) -> int:
        """
        일관성 있는 응답의 수를 계산합니다.

        Args:
            responses (List[str]): 응답 리스트

        Returns:
            int: 일관된 응답 수
        """
        if len(responses) < 2:
            return len(responses)

        # 의미적 유사도 기반 클러스터링
        embeddings = self.sentence_model.encode(responses)
        similarity_matrix = cosine_similarity(embeddings)

        # 각 응답이 얼마나 많은 다른 응답과 유사한지 계산
        consistent_count = 0
        for i, row in enumerate(similarity_matrix):
            similar_responses = sum(1 for j, sim in enumerate(row) if i != j and sim >= self.similarity_threshold)
            # 절반 이상의 다른 응답과 유사하면 일관된 것으로 간주
            if similar_responses >= (len(responses) - 1) / 2:
                consistent_count += 1

        return consistent_count

    def _calculate_average_similarity(self, responses: List[str]) -> float:
        """
        응답들 간의 평균 유사도를 계산합니다.

        Args:
            responses (List[str]): 응답 리스트

        Returns:
            float: 평균 유사도
        """
        if len(responses) < 2:
            return 1.0

        embeddings = self.sentence_model.encode(responses)
        similarity_matrix = cosine_similarity(embeddings)

        # 대각선 제외 평균
        np.fill_diagonal(similarity_matrix, 0)
        n = len(responses)
        avg_similarity = np.sum(similarity_matrix) / (n * (n - 1))

        return float(avg_similarity)

    def _calculate_confidence_from_consistency(
        self,
        consistency_rate: float,
        num_consistent: int,
        total_responses: int
    ) -> float:
        """
        일관성 지표들로부터 최종 신뢰도를 계산합니다.

        Args:
            consistency_rate (float): 일관성 비율
            num_consistent (int): 일관된 응답 수
            total_responses (int): 전체 응답 수

        Returns:
            float: 최종 신뢰도 (0.0 ~ 1.0)
        """
        # 일관성 비율 (50%)
        consistency_score = consistency_rate

        # 일관된 응답 비율 (50%)
        consistent_ratio = num_consistent / total_responses if total_responses > 0 else 0.0

        # 가중 평균
        confidence = 0.5 * consistency_score + 0.5 * consistent_ratio

        return round(confidence, 3)

    def _extract_topic_from_text(self, text: str) -> str:
        """
        텍스트에서 주제를 추출합니다.

        Args:
            text (str): 텍스트

        Returns:
            str: 추출된 주제
        """
        # 제목이나 첫 문장에서 주제 추출 시도
        lines = text.split('\n')
        for line in lines:
            if line.strip() and len(line.strip()) > 10:
                # 제목 형식 확인
                if line.startswith('제목:') or line.startswith('#'):
                    return line.replace('제목:', '').replace('#', '').strip()

        # 첫 문장 반환
        first_sentence = text.split('.')[0] if '.' in text else text[:100]
        return first_sentence.strip()