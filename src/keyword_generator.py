"""
키워드 생성 모듈
LLM을 활용하여 주제 기반 키워드를 자동 생성
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    logger.error("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai' 실행")
    raise

from src.config import config


@dataclass
class KeywordResult:
    """키워드 생성 결과를 담는 데이터 클래스"""
    topic: str
    primary_keywords: List[str]
    related_terms: List[str]
    synonyms: List[str]
    context_keywords: List[str]
    confidence_score: float
    generation_time: float
    raw_response: str


class KeywordGenerator:
    """LLM 기반 키워드 생성기"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or config.get_openai_api_key()
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다")

        self.client = AsyncOpenAI(api_key=self.api_key)

        # 설정값 로드
        self.max_retries = config.get_max_retry_count()
        self.timeout = config.get_keyword_generation_timeout()
        self.temperature = config.get_openai_temperature()
        self.max_tokens = config.get_openai_max_tokens()

        logger.info(f"KeywordGenerator 초기화 완료 (모델: {self.model})")

    async def generate_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        num_keywords: int = 20
    ) -> KeywordResult:
        """주제에 대한 키워드를 생성합니다"""
        start_time = time.time()
        logger.info(f"키워드 생성 시작: '{topic}' (모델: {self.model})")

        if not topic or not topic.strip():
            raise ValueError("주제가 비어있습니다")

        topic = topic.strip()

        try:
            prompt = self._build_prompt(topic, context, num_keywords)
            raw_response = await self._call_llm(prompt)
            keyword_result = self._parse_response(topic, raw_response, time.time() - start_time)

            logger.success(
                f"키워드 생성 완료: {len(keyword_result.primary_keywords)}개 핵심 키워드, "
                f"신뢰도 {keyword_result.confidence_score:.2f}, "
                f"소요시간 {keyword_result.generation_time:.1f}초"
            )

            return keyword_result

        except Exception as e:
            logger.error(f"키워드 생성 실패: {str(e)}")
            raise

    def _build_prompt(self, topic: str, context: Optional[str], num_keywords: int) -> str:
        """키워드 생성을 위한 프롬프트 구성"""
        base_prompt = f"""주제 "{topic}"에 대한 이슈 모니터링을 위한 키워드를 생성해주세요.

요구사항:
1. 핵심 키워드 (Primary Keywords): 주제와 직접적으로 관련된 핵심 용어들
2. 관련 용어 (Related Terms): 주제와 연관된 기술, 개념, 트렌드
3. 동의어 (Synonyms): 같은 의미의 다른 표현들
4. 맥락 키워드 (Context Keywords): 해당 분야의 배경 지식이나 관련 영역

각 카테고리별로 {max(3, num_keywords//4)}~{min(8, num_keywords//2)}개씩 생성해주세요."""

        if context:
            base_prompt += f"\n\n추가 맥락: {context}"

        base_prompt += """

응답 형식 (반드시 유효한 JSON으로 응답):
{
    "primary_keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
    "related_terms": ["용어1", "용어2", "용어3", "용어4"],
    "synonyms": ["동의어1", "동의어2", "동의어3"],
    "context_keywords": ["맥락1", "맥락2", "맥락3", "맥락4"],
    "confidence": 0.9
}

주의사항:
- 키워드는 한국어와 영어를 모두 포함해주세요
- 검색에 효과적인 구체적인 용어를 선택해주세요  
- 너무 일반적이거나 모호한 용어는 피해주세요
- confidence는 생성 품질에 대한 자신감을 0.0-1.0 사이로 표현해주세요
- 반드시 위의 JSON 형식으로만 응답해주세요"""

        return base_prompt

    async def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM API 호출 시도 {attempt + 1}/{self.max_retries}")

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 이슈 모니터링을 위한 키워드 생성 전문가입니다. "
                                     "주어진 주제에 대해 포괄적이고 효과적인 검색 키워드를 생성해주세요. "
                                     "반드시 유효한 JSON 형식으로만 응답하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM 응답이 비어있습니다")

                return content.strip()

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"LLM API 호출 실패 (시도 {attempt + 1}): {error_msg}")

                if attempt == self.max_retries - 1:
                    if "401" in error_msg:
                        raise ValueError("OpenAI API 키가 유효하지 않습니다.")
                    elif "429" in error_msg:
                        raise ValueError("API 사용량 한도를 초과했습니다.")
                    elif "quota" in error_msg.lower():
                        raise ValueError("OpenAI 크레딧이 부족합니다.")
                    else:
                        raise ValueError(f"LLM API 호출 최종 실패: {error_msg}")

                # 지수 백오프 대기
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

    def _parse_response(self, topic: str, raw_response: str, generation_time: float) -> KeywordResult:
        """LLM 응답을 파싱하여 KeywordResult로 변환"""
        try:
            # JSON 추출
            cleaned_response = re.sub(r'```json\s*\n', '', raw_response)
            cleaned_response = re.sub(r'\n\s*```', '', cleaned_response)

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            if not json_match:
                raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")

            json_str = json_match.group()

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 파싱 실패: {e}")
                raise ValueError(f"JSON 파싱 실패: {e}")

            # 필수 필드 검증
            required_fields = ['primary_keywords', 'related_terms', 'synonyms', 'context_keywords']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"필수 필드가 누락되었습니다: {missing_fields}")

            # 데이터 정제
            primary_keywords = self._clean_keywords(data['primary_keywords'])
            related_terms = self._clean_keywords(data['related_terms'])
            synonyms = self._clean_keywords(data['synonyms'])
            context_keywords = self._clean_keywords(data['context_keywords'])

            # 신뢰도 점수 처리
            raw_confidence = float(data.get('confidence', 0.8))
            confidence_score = max(0.0, min(1.0, raw_confidence))

            # 최소 키워드 보장
            if len(primary_keywords) == 0:
                primary_keywords = [topic]
                confidence_score = 0.5

            return KeywordResult(
                topic=topic,
                primary_keywords=primary_keywords,
                related_terms=related_terms,
                synonyms=synonyms,
                context_keywords=context_keywords,
                confidence_score=confidence_score,
                generation_time=generation_time,
                raw_response=raw_response
            )

        except Exception as e:
            logger.error(f"응답 파싱 실패: {str(e)}")
            return self._create_fallback_result(topic, raw_response, generation_time)

    def _clean_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 리스트 정제"""
        if not isinstance(keywords, list):
            logger.warning(f"키워드가 리스트가 아닙니다: {type(keywords)}")
            return []

        cleaned = []
        for keyword in keywords:
            if isinstance(keyword, str):
                keyword = keyword.strip().strip('"\'').strip()
                if keyword and len(keyword) > 1:
                    cleaned.append(keyword)
            elif keyword is not None:
                keyword_str = str(keyword).strip()
                if keyword_str and len(keyword_str) > 1:
                    cleaned.append(keyword_str)

        # 중복 제거
        seen = set()
        unique_keywords = []
        for keyword in cleaned:
            lower_keyword = keyword.lower()
            if lower_keyword not in seen:
                seen.add(lower_keyword)
                unique_keywords.append(keyword)

        return unique_keywords[:12]  # 최대 12개로 제한

    def _create_fallback_result(self, topic: str, raw_response: str, generation_time: float) -> KeywordResult:
        """파싱 실패시 기본 키워드 결과 생성"""
        logger.warning("파싱 실패로 인한 폴백 키워드 생성")

        basic_keywords = [topic.strip()]
        words = topic.split()
        if len(words) > 1:
            basic_keywords.extend([word.strip() for word in words if len(word.strip()) > 1])

        basic_keywords = list(dict.fromkeys(basic_keywords))

        return KeywordResult(
            topic=topic,
            primary_keywords=basic_keywords,
            related_terms=[],
            synonyms=[],
            context_keywords=[],
            confidence_score=0.2,
            generation_time=generation_time,
            raw_response=raw_response
        )

    def get_all_keywords(self, result: KeywordResult) -> List[str]:
        """모든 키워드를 하나의 리스트로 반환"""
        all_keywords = []
        all_keywords.extend(result.primary_keywords)
        all_keywords.extend(result.related_terms)
        all_keywords.extend(result.synonyms)
        all_keywords.extend(result.context_keywords)
        return list(dict.fromkeys(all_keywords))

    def format_keywords_summary(self, result: KeywordResult) -> str:
        """키워드 결과를 요약 문자열로 포맷팅"""
        total_count = len(self.get_all_keywords(result))
        confidence_percent = int(result.confidence_score * 100)

        summary = f"**키워드 생성 완료** (주제: {result.topic})\n"
        summary += f"📊 총 {total_count}개 키워드 | 신뢰도: {confidence_percent}% | 소요시간: {result.generation_time:.1f}초\n\n"

        if result.primary_keywords:
            summary += f"🎯 **핵심**: {', '.join(result.primary_keywords[:5])}"
            if len(result.primary_keywords) > 5:
                summary += f" 외 {len(result.primary_keywords) - 5}개"
            summary += "\n"

        if result.related_terms:
            summary += f"🔗 **관련**: {', '.join(result.related_terms[:3])}"
            if len(result.related_terms) > 3:
                summary += f" 외 {len(result.related_terms) - 3}개"
            summary += "\n"

        return summary


# 편의 함수들
def create_keyword_generator(api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> KeywordGenerator:
    """키워드 생성기 인스턴스 생성"""
    return KeywordGenerator(api_key=api_key, model=model)


async def generate_keywords_for_topic(topic: str, context: Optional[str] = None) -> KeywordResult:
    """주제에 대한 키워드를 생성하는 편의 함수"""
    generator = create_keyword_generator()
    return await generator.generate_keywords(topic, context)


if __name__ == "__main__":
    print("🧪 키워드 생성기 테스트")
    print("pytest로 테스트를 실행하세요:")
    print("pytest tests/test_keyword_generator.py -v")