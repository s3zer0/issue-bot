"""
키워드 생성 모듈
LLM을 활용하여 주제 기반 키워드를 자동 생성
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    from openai import AsyncOpenAI, AuthenticationError
except ImportError:
    logger.error("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행해주세요.")
    class AsyncOpenAI: pass
    class AuthenticationError(Exception): pass

from src.config import config


@dataclass
class KeywordResult:
    """키워드 생성 결과를 담는 데이터 클래스 (synonyms 필드 제거)"""
    topic: str
    primary_keywords: List[str]
    related_terms: List[str]
    context_keywords: List[str]
    confidence_score: float
    generation_time: float
    raw_response: str


class KeywordGenerator:
    """LLM 기반 키워드 생성기"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.get_openai_api_key()
        self.model = model or config.get_openai_model()

        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.max_retries = config.get_max_retry_count()
        self.timeout = config.get_keyword_generation_timeout()
        self.temperature = config.get_openai_temperature()
        self.max_tokens = config.get_openai_max_tokens()
        logger.info(f"KeywordGenerator 초기화 완료 (모델: {self.model})")

    async def generate_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        num_keywords: int = 15
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
            logger.error(f"키워드 생성 실패: {e}")
            raise

    def _build_prompt(self, topic: str, context: Optional[str], num_keywords: int) -> str:
        """
        [수정됨] 키워드 생성을 위한 프롬프트를 개선하여 단순 번역을 방지하고 품질을 높입니다.
        """
        base_prompt = f"""주제 "{topic}"에 대한 심층적인 이슈 모니터링을 위한 검색 키워드를 생성해주세요.

**요구사항:**
1.  **핵심 키워드 (Primary Keywords)**: 주제를 가장 잘 나타내는 핵심 단어 및 구문. (예: '인공지능', 'AI', 'Generative AI')
2.  **관련 용어 (Related Terms)**: 주제와 밀접하게 연관된 하위 기술, 주요 인물, 관련 제품/서비스, 주요 기업/기관 이름 등 구체적인 용어. (예: 'LLM', 'OpenAI', 'Sora', 'Figure 01')
3.  **맥락 키워드 (Context Keywords)**: 주제가 포함된 더 넓은 산업 분야나 사회적 맥락을 나타내는 용어. (예: '디지털 전환', '노동 시장 변화', 'AI 윤리')

**생성 가이드라인:**
-   각 카테고리별로 {max(3, num_keywords//3)}~{min(6, num_keywords//2)}개씩 생성해주세요.
-   한국어와 영어를 자연스럽게 혼용하되, 실제 업계에서 통용되는 용어를 사용해주세요 (예: '딥페이크'는 한글로, 'LLM'은 영어로).
-   **단순 번역을 절대 피해주세요.** 예를 들어 'AI'와 '인공지능'을 모두 생성하기보다, '생성형 AI', 'AGI' 등 더 구체적이거나 다른 차원의 키워드를 제안해야 합니다.
-   검색 엔진에서 유의미한 결과를 얻을 수 있는 구체적이고 전문적인 용어를 선호합니다."""

        if context:
            base_prompt += f"\n\n**추가 맥락**: {context}"

        base_prompt += """

**응답 형식 (반드시 유효한 JSON으로만 응답):**
{
    "primary_keywords": ["키워드1", "키워드2"],
    "related_terms": ["관련용어1", "관련용어2", "관련용어3"],
    "context_keywords": ["맥락1", "맥락2", "맥락3", "맥락4"],
    "confidence": 0.9
}"""
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
                            "content": "당신은 특정 주제에 대한 깊이 있는 분석을 위해 검색 키워드를 생성하는 IT 전문 분석가입니다. 반드시 유효한 JSON 형식으로만 응답해야 합니다."
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
            except AuthenticationError as e:
                logger.error(f"OpenAI 인증 오류: {e.message}")
                raise ValueError("OpenAI API 키가 유효하지 않습니다.") from e
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"LLM API 호출 실패 (시도 {attempt + 1}): {error_msg}")
                if attempt == self.max_retries - 1:
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        raise ValueError("API 사용량 한도를 초과했습니다.")
                    elif "quota" in error_msg.lower():
                        raise ValueError("OpenAI 크레딧이 부족합니다.")
                    else:
                        raise ValueError(f"LLM API 호출 최종 실패: {error_msg}")
                await asyncio.sleep(2 ** attempt)
        raise ValueError("모든 재시도에 실패했습니다.")


    def _parse_response(self, topic: str, raw_response: str, generation_time: float) -> KeywordResult:
        """LLM 응답을 파싱하여 KeywordResult로 변환"""
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            if not json_match:
                raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")

            data = json.loads(json_match.group())

            # [수정됨] 'synonyms' 필드 제거
            required_fields = ['primary_keywords', 'related_terms', 'context_keywords']
            if not all(field in data for field in required_fields):
                raise ValueError(f"필수 필드가 누락되었습니다: {required_fields}")

            primary_keywords = self._clean_keywords(data.get('primary_keywords', []))
            related_terms = self._clean_keywords(data.get('related_terms', []))
            context_keywords = self._clean_keywords(data.get('context_keywords', []))
            confidence_score = min(1.0, max(0.0, float(data.get('confidence', 0.8))))

            if not primary_keywords:
                primary_keywords = [topic]
                confidence_score = 0.5

            return KeywordResult(
                topic=topic,
                primary_keywords=primary_keywords,
                related_terms=related_terms,
                context_keywords=context_keywords,
                confidence_score=confidence_score,
                generation_time=generation_time,
                raw_response=raw_response
            )
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return self._create_fallback_result(topic, raw_response, generation_time)

    def _clean_keywords(self, keywords: List[Any]) -> List[str]:
        """키워드 리스트를 정제합니다."""
        if not isinstance(keywords, list):
            logger.warning(f"키워드 데이터가 리스트가 아닙니다: {type(keywords)}")
            return []

        cleaned = []
        for keyword in keywords:
            if isinstance(keyword, str):
                keyword = keyword.strip().strip('"\'')
                if len(keyword) > 1:
                    cleaned.append(keyword)

        seen = set()
        unique_keywords = []
        for keyword in cleaned:
            lower_keyword = keyword.lower()
            if lower_keyword not in seen:
                seen.add(lower_keyword)
                unique_keywords.append(keyword)
        return unique_keywords[:12]

    def _create_fallback_result(self, topic: str, raw_response: str, generation_time: float) -> KeywordResult:
        """파싱 실패 시 기본 키워드 결과 생성"""
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
            context_keywords=[],
            confidence_score=0.2,
            generation_time=generation_time,
            raw_response=raw_response
        )

    def get_all_keywords(self, result: KeywordResult) -> List[str]:
        """[수정됨] 모든 키워드를 하나의 리스트로 반환"""
        all_keywords = (result.primary_keywords + result.related_terms + result.context_keywords)
        return list(dict.fromkeys(all_keywords))

    def format_keywords_summary(self, result: KeywordResult) -> str:
        """키워드 결과를 요약 문자열로 포맷팅"""
        total_count = len(self.get_all_keywords(result))
        confidence_percent = int(result.confidence_score * 100)

        summary = (f"**키워드 생성 완료** (주제: {result.topic})\n"
                   f"📊 총 {total_count}개 키워드 | 신뢰도: {confidence_percent}% | 소요시간: {result.generation_time:.1f}초\n\n")

        if result.primary_keywords:
            keywords_str = ', '.join(result.primary_keywords[:5])
            extra_count = len(result.primary_keywords) - 5
            summary += f"🎯 **핵심**: {keywords_str}"
            if extra_count > 0:
                summary += f" 외 {extra_count}개"
            summary += "\n"

        if result.related_terms:
            keywords_str = ', '.join(result.related_terms[:4])
            extra_count = len(result.related_terms) - 4
            summary += f"🔗 **관련**: {keywords_str}"
            if extra_count > 0:
                summary += f" 외 {extra_count}개"
            summary += "\n"
        return summary

def create_keyword_generator(api_key: Optional[str] = None, model: Optional[str] = None) -> KeywordGenerator:
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