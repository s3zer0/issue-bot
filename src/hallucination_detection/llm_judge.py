"""
LLM-as-a-Judge 기반 환각 탐지기 - 완전 수정 버전.

GPT-4o를 활용하여 텍스트의 사실성, 논리성, 일관성을 평가하고
환각 가능성을 판단합니다.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from openai import AsyncOpenAI

from src.config import config
from .base import BaseHallucinationDetector
from .models import HallucinationScore


@dataclass
class LLMJudgeScore(HallucinationScore):
    """LLM Judge 분석 결과."""

    category_scores: Dict[str, float] = field(default_factory=dict)
    problematic_areas: List[Dict[str, str]] = field(default_factory=list)
    judge_reasoning: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.method_name = "LLM-Judge"

    def get_summary(self) -> str:
        """LLM Judge 분석 결과 요약."""
        if self.category_scores:
            categories = ", ".join([
                f"{cat}: {score:.2f}"
                for cat, score in self.category_scores.items()
            ])
            return (
                f"LLM Judge - 신뢰도: {self.confidence:.2f} "
                f"({categories})"
            )
        else:
            return f"LLM Judge - 신뢰도: {self.confidence:.2f}"


class LLMJudgeDetector(BaseHallucinationDetector):
    """LLM을 심판으로 활용한 환각 탐지기."""

    def __init__(self, model_name: str = "gpt-4o"):
        """
        LLM Judge 탐지기를 초기화합니다.

        Args:
            model_name (str): 평가에 사용할 OpenAI 모델
        """
        super().__init__("LLM-Judge")

        # OpenAI 클라이언트 설정
        self.api_key = config.get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = str(model_name)  # 명시적 str 변환

        # 평가 카테고리별 가중치
        self.category_weights = {
            "factual_accuracy": 0.35,
            "logical_consistency": 0.25,
            "contextual_relevance": 0.20,
            "source_reliability": 0.20
        }

        self.is_initialized = True
        logger.info(f"LLM Judge 탐지기 초기화 완료 (모델: {self.model})")

    async def analyze_text(self, text: str, context: Optional[str] = None) -> LLMJudgeScore:
        """
        LLM을 사용하여 텍스트의 환각 가능성을 평가합니다.

        Args:
            text (str): 분석할 텍스트
            context (Optional[str]): 텍스트의 맥락 (주제, 질문 등)

        Returns:
            LLMJudgeScore: 분석 결과
        """
        logger.info(f"LLM Judge 분석 시작 (텍스트 길이: {len(text)})")

        try:
            # 평가 프롬프트 생성
            evaluation_prompt = self._create_evaluation_prompt(text, context)

            # LLM에 평가 요청
            evaluation_result = await self._get_llm_evaluation(evaluation_prompt)

            # 결과 파싱 및 점수 계산
            scores = self._parse_evaluation_result(evaluation_result)
            final_confidence = self._calculate_weighted_score(scores)

            # 환각 가능성이 높은 부분 식별
            problematic_areas = self._identify_problematic_areas(evaluation_result)

            logger.info(f"LLM Judge 분석 완료 - 신뢰도: {final_confidence:.2f}")

            return LLMJudgeScore(
                confidence=final_confidence,
                category_scores=scores,
                problematic_areas=problematic_areas,
                judge_reasoning=evaluation_result.get("overall_reasoning", ""),
                analysis_details={
                    "model_used": str(self.model),  # 🔧 명시적 str 변환
                    "context_provided": bool(context is not None),  # 🔧 명시적 bool 변환
                    "text_length": int(len(text)),  # 🔧 명시적 int 변환
                    "evaluation_success": True
                }
            )

        except Exception as e:
            logger.error(f"LLM Judge 분석 실패: {e}")
            # 오류 시 중립적인 점수 반환
            return LLMJudgeScore(
                confidence=0.5,
                category_scores={
                    "factual_accuracy": 0.5,
                    "logical_consistency": 0.5,
                    "contextual_relevance": 0.5,
                    "source_reliability": 0.5
                },
                problematic_areas=[],
                judge_reasoning=f"평가 중 오류 발생: {str(e)}",
                analysis_details={
                    "error": str(e),
                    "model_used": str(self.model),
                    "evaluation_success": False
                }
            )

    def _create_evaluation_prompt(self, text: str, context: Optional[str]) -> str:
        """
        LLM 평가를 위한 구조화된 프롬프트를 생성합니다.

        Args:
            text (str): 평가할 텍스트
            context (Optional[str]): 맥락 정보

        Returns:
            str: 평가 프롬프트
        """
        context_section = f"원래 질문/주제: {context}\n\n" if context else ""

        prompt = f"""당신은 텍스트의 신뢰성을 평가하는 전문가입니다. 다음 텍스트를 분석하여 환각(hallucination) 가능성을 평가해주세요.

{context_section}평가할 텍스트:
{text}

다음 4가지 카테고리에 대해 각각 0-100점으로 평가하고, 구체적인 이유를 제시해주세요:

1. **사실적 정확성 (Factual Accuracy)**
   - 언급된 사실, 숫자, 날짜, 이름 등이 정확한가?
   - 검증 가능한 정보인가?
   - 일반적으로 알려진 사실과 일치하는가?

2. **논리적 일관성 (Logical Consistency)**
   - 텍스트 내에서 모순되는 주장이 있는가?
   - 인과관계가 논리적인가?
   - 주장과 근거가 일치하는가?

3. **맥락적 관련성 (Contextual Relevance)**
   - 주어진 맥락과 관련이 있는가?
   - 주제에서 벗어나지 않는가?
   - 적절한 수준의 세부사항을 포함하는가?

4. **출처 신뢰성 (Source Reliability)**
   - 언급된 출처나 정보가 신뢰할 만한가?
   - 권위있는 기관이나 전문가의 정보인가?
   - 검증 가능한 출처가 제시되었는가?

**응답 형식 (반드시 유효한 JSON으로 응답):**
{{
  "factual_accuracy": {{
    "score": 점수(0-100),
    "reasoning": "평가 이유"
  }},
  "logical_consistency": {{
    "score": 점수(0-100),
    "reasoning": "평가 이유"
  }},
  "contextual_relevance": {{
    "score": 점수(0-100),
    "reasoning": "평가 이유"
  }},
  "source_reliability": {{
    "score": 점수(0-100),
    "reasoning": "평가 이유"
  }},
  "problematic_areas": [
    {{
      "text": "문제가 있는 텍스트 부분",
      "issue": "구체적인 문제점"
    }}
  ],
  "overall_reasoning": "종합적인 평가 이유"
}}

중요: 반드시 위 JSON 형식으로만 응답하세요."""

        return prompt

    async def _get_llm_evaluation(self, prompt: str) -> Dict:
        """
        OpenAI API를 호출하여 LLM 평가를 수행합니다.

        Args:
            prompt (str): 평가 프롬프트

        Returns:
            Dict: 평가 결과

        Raises:
            Exception: API 호출 실패 시
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # 🔧 JSON 직렬화 문제 방지를 위한 안전한 파라미터 구성
                request_params = {
                    "model": str(self.model),  # 명시적 str 변환
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 텍스트의 신뢰성을 평가하는 전문가입니다. 반드시 유효한 JSON 형식으로만 응답하세요."
                        },
                        {
                            "role": "user",
                            "content": str(prompt)  # 명시적 str 변환
                        }
                    ],
                    "temperature": 0.1,  # 일관된 평가를 위해 낮은 temperature
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}  # JSON 출력 강제
                }

                logger.debug(f"LLM Judge API 호출 시도 {attempt + 1}/{max_retries}")

                response = await self.client.chat.completions.create(**request_params)
                content = response.choices[0].message.content

                if not content:
                    raise ValueError("LLM 응답이 비어있습니다")

                # JSON 파싱 시도
                try:
                    result = json.loads(content)
                    logger.debug("JSON 파싱 성공")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패 (시도 {attempt + 1}), 백업 파싱 시도: {e}")
                    return self._fallback_parse(content)

            except Exception as e:
                logger.error(f"LLM 평가 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")

                if attempt == max_retries - 1:
                    # 최종 실패 시 기본값 반환
                    logger.error("모든 재시도 실패, 기본값 반환")
                    return self._get_default_evaluation()

                # 재시도 전 잠시 대기
                import asyncio
                await asyncio.sleep(1)

        return self._get_default_evaluation()

    def _get_default_evaluation(self) -> Dict:
        """API 호출 실패 시 사용할 기본 평가 결과."""
        return {
            "factual_accuracy": {"score": 50, "reasoning": "API 호출 실패로 기본값 사용"},
            "logical_consistency": {"score": 50, "reasoning": "API 호출 실패로 기본값 사용"},
            "contextual_relevance": {"score": 50, "reasoning": "API 호출 실패로 기본값 사용"},
            "source_reliability": {"score": 50, "reasoning": "API 호출 실패로 기본값 사용"},
            "problematic_areas": [],
            "overall_reasoning": "API 호출 실패로 인해 중립적 점수 적용"
        }

    def _fallback_parse(self, content: str) -> Dict:
        """
        JSON 파싱 실패 시 백업 파싱 방법.

        Args:
            content (str): LLM 응답

        Returns:
            Dict: 파싱된 결과 (기본값 포함)
        """
        logger.warning("백업 파싱 메서드 사용")

        # 기본 응답 구조
        result = {
            "factual_accuracy": {"score": 50, "reasoning": "파싱 실패로 기본값 사용"},
            "logical_consistency": {"score": 50, "reasoning": "파싱 실패로 기본값 사용"},
            "contextual_relevance": {"score": 50, "reasoning": "파싱 실패로 기본값 사용"},
            "source_reliability": {"score": 50, "reasoning": "파싱 실패로 기본값 사용"},
            "problematic_areas": [],
            "overall_reasoning": "응답 파싱에 실패하여 중립적 점수 적용"
        }

        # 간단한 키워드 기반 점수 조정
        content_lower = content.lower()

        # 긍정적 키워드
        positive_keywords = ['정확', 'correct', 'accurate', '신뢰', 'reliable', '논리적', 'logical']
        negative_keywords = ['부정확', 'incorrect', 'wrong', '모순', 'contradiction', '의심', 'doubt']

        positive_count = sum(1 for word in positive_keywords if word in content_lower)
        negative_count = sum(1 for word in negative_keywords if word in content_lower)

        # 점수 조정
        if positive_count > negative_count:
            adjustment = 20  # 70점으로 조정
        elif negative_count > positive_count:
            adjustment = -20  # 30점으로 조정
        else:
            adjustment = 0  # 50점 유지

        for category in result:
            if isinstance(result[category], dict) and "score" in result[category]:
                result[category]["score"] = max(0, min(100, 50 + adjustment))

        return result

    def _parse_evaluation_result(self, result: Dict) -> Dict[str, float]:
        """
        평가 결과에서 카테고리별 점수를 추출합니다.

        Args:
            result (Dict): LLM 평가 결과

        Returns:
            Dict[str, float]: 카테고리별 점수 (0.0 ~ 1.0)
        """
        scores = {}

        for category in self.category_weights.keys():
            if category in result and isinstance(result[category], dict):
                # 100점 만점을 0-1 범위로 정규화
                raw_score = result[category].get("score", 50)
                # 안전한 범위 보장
                normalized_score = max(0.0, min(1.0, float(raw_score) / 100.0))
                scores[category] = normalized_score
            else:
                # 카테고리가 없으면 중립 점수
                scores[category] = 0.5
                logger.warning(f"평가 결과에 {category} 카테고리가 없습니다.")

        return scores

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        카테고리별 점수를 가중 평균하여 최종 신뢰도를 계산합니다.

        Args:
            scores (Dict[str, float]): 카테고리별 점수

        Returns:
            float: 최종 신뢰도 (0.0 ~ 1.0)
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for category, weight in self.category_weights.items():
            if category in scores:
                weighted_sum += float(scores[category]) * float(weight)
                total_weight += float(weight)

        if total_weight > 0:
            result = weighted_sum / total_weight
            # 안전한 범위 보장 및 반올림
            return round(max(0.0, min(1.0, result)), 3)
        else:
            return 0.5  # 기본값

    def _identify_problematic_areas(self, result: Dict) -> List[Dict[str, str]]:
        """
        평가 결과에서 문제가 있는 부분을 추출합니다.

        Args:
            result (Dict): LLM 평가 결과

        Returns:
            List[Dict[str, str]]: 문제 영역 리스트
        """
        problematic_areas = result.get("problematic_areas", [])

        # 유효성 검사 및 안전한 변환
        valid_areas = []
        for area in problematic_areas:
            try:
                if isinstance(area, dict) and "text" in area and "issue" in area:
                    valid_areas.append({
                        "text": str(area["text"])[:200],  # 최대 200자, 명시적 str 변환
                        "issue": str(area["issue"])[:500]  # 최대 500자, 명시적 str 변환
                    })
            except Exception as e:
                logger.warning(f"문제 영역 파싱 실패: {e}")
                continue

        return valid_areas[:10]  # 최대 10개로 제한

    async def analyze_claims(self, claims: List[str], context: Optional[str] = None) -> Dict[str, float]:
        """
        개별 주장들의 신뢰도를 평가합니다.

        Args:
            claims (List[str]): 평가할 주장 리스트
            context (Optional[str]): 맥락 정보

        Returns:
            Dict[str, float]: 각 주장의 신뢰도 점수
        """
        claim_scores = {}

        for claim in claims:
            try:
                score = await self.analyze_text(claim, context)
                claim_scores[claim] = score.confidence
            except Exception as e:
                logger.warning(f"주장 분석 실패: {claim[:50]}... - {e}")
                claim_scores[claim] = 0.5  # 기본값

        return claim_scores

    def get_detector_info(self) -> Dict[str, str]:
        """탐지기 정보 반환."""
        return {
            "name": self.name,
            "model": str(self.model),
            "is_initialized": str(self.is_initialized),
            "api_key_set": str(bool(self.api_key))
        }