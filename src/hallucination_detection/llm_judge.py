"""
LLM-as-a-Judge 기반 환각 탐지기.

GPT-4o를 활용하여 텍스트의 사실성, 논리성, 일관성을 평가하고
환각 가능성을 판단합니다.
"""

import json
from typing import Dict, List, Optional, Tuple
from loguru import logger
from openai import AsyncOpenAI

from src.config import config
from .base import BaseHallucinationDetector
from .models import HallucinationScore


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
        self.model = model_name

        # 평가 카테고리별 가중치
        self.category_weights = {
            "factual_accuracy": 0.35,
            "logical_consistency": 0.25,
            "contextual_relevance": 0.20,
            "source_reliability": 0.20
        }

        self.is_initialized = True
        logger.info(f"LLM Judge 탐지기 초기화 완료 (모델: {self.model})")

    async def analyze_text(self, text: str, context: Optional[str] = None) -> HallucinationScore:
        """
        LLM을 사용하여 텍스트의 환각 가능성을 평가합니다.

        Args:
            text (str): 분석할 텍스트
            context (Optional[str]): 텍스트의 맥락 (주제, 질문 등)

        Returns:
            HallucinationScore: 분석 결과
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
                    "model_used": self.model,
                    "context_provided": context is not None,
                    "text_length": len(text)
                }
            )

        except Exception as e:
            logger.error(f"LLM Judge 분석 실패: {e}")
            # 오류 시 중립적인 점수 반환
            return LLMJudgeScore(
                confidence=0.5,
                category_scores={},
                problematic_areas=[],
                judge_reasoning=f"평가 중 오류 발생: {str(e)}",
                analysis_details={"error": str(e)}
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
   - 주제/질문에 적절히 답하고 있는가?
   - 불필요한 정보나 주제 이탈이 있는가?
   - 맥락에 맞는 적절한 수준의 정보를 제공하는가?

4. **출처 신뢰성 (Source Reliability)**
   - 구체적인 출처가 언급되었는가?
   - 출처가 신뢰할 만한가?
   - 주장에 대한 근거가 제시되었는가?

추가로 다음 사항을 포함해주세요:
- **환각 가능성이 높은 부분**: 구체적으로 어떤 문장이나 주장이 의심스러운지
- **전반적인 평가**: 종합적인 신뢰도와 그 이유

응답은 반드시 다음 JSON 형식으로 제공해주세요:
{{
    "factual_accuracy": {{
        "score": 0-100,
        "reasoning": "평가 이유"
    }},
    "logical_consistency": {{
        "score": 0-100,
        "reasoning": "평가 이유"
    }},
    "contextual_relevance": {{
        "score": 0-100,
        "reasoning": "평가 이유"
    }},
    "source_reliability": {{
        "score": 0-100,
        "reasoning": "평가 이유"
    }},
    "problematic_areas": [
        {{
            "text": "의심스러운 문장/구절",
            "issue": "문제점 설명"
        }}
    ],
    "overall_reasoning": "전반적인 평가와 종합적인 판단"
}}"""

        return prompt

    async def _get_llm_evaluation(self, prompt: str) -> Dict:
        """
        LLM에 평가를 요청하고 결과를 받습니다.

        Args:
            prompt (str): 평가 프롬프트

        Returns:
            Dict: 평가 결과
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 텍스트의 신뢰성과 환각 가능성을 평가하는 전문가입니다. 객관적이고 구체적인 근거를 바탕으로 평가해주세요."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 일관된 평가를 위해 낮은 temperature
                max_tokens=2000,
                response_format={"type": "json_object"}  # JSON 응답 강제
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            raise ValueError("LLM 응답을 파싱할 수 없습니다.")
        except Exception as e:
            logger.error(f"LLM 평가 요청 실패: {e}")
            raise

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
                scores[category] = raw_score / 100.0
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
                weighted_sum += scores[category] * weight
                total_weight += weight

        if total_weight > 0:
            return round(weighted_sum / total_weight, 3)
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

        # 유효성 검사
        valid_areas = []
        for area in problematic_areas:
            if isinstance(area, dict) and "text" in area and "issue" in area:
                valid_areas.append({
                    "text": str(area["text"])[:200],  # 최대 200자
                    "issue": str(area["issue"])[:500]  # 최대 500자
                })

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
            score = await self.analyze_text(claim, context)
            claim_scores[claim] = score.confidence

        return claim_scores


from dataclasses import dataclass, field
from typing import Dict, List


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
        categories = ", ".join([
            f"{cat}: {score:.2f}"
            for cat, score in self.category_scores.items()
        ])

        return (
            f"LLM Judge - 신뢰도: {self.confidence:.2f} "
            f"({categories})"
        )