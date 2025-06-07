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
from src.models import KeywordResult # 중앙 데이터 모델 import


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
        num_keywords: int = 50
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
        모든 주제에 대해 전문적이고 기술적인 키워드 생성을 위한 상세한 범용 프롬프트
        """

        base_prompt = f"""주제 "{topic}"에 대한 전문적이고 심층적인 이슈 모니터링을 위한 고품질 검색 키워드를 생성해주세요.

**목적**: 생성된 키워드는 최신 뉴스, 기술 문서, 연구 논문, 업계 보고서에서 해당 주제와 관련된 가장 중요하고 시의적절한 정보를 찾는 데 사용됩니다.

**키워드 카테고리 상세 설명:**

1. **핵심 키워드 (Primary Keywords)** - {max(5, num_keywords // 3)}~{min(10, num_keywords // 2)}개:
   - 주제의 본질을 정확히 포착하는 가장 중요한 전문 용어
   - 해당 분야 전문가들이 일상적으로 사용하는 표준 용어
   - 학술 논문, 기술 문서, 업계 리포트의 제목이나 키워드 섹션에 자주 등장하는 용어
   - 검색 엔진에서 높은 정확도로 관련 콘텐츠를 찾을 수 있는 구별력 있는 용어
   - 약어와 전체 명칭 중 업계에서 더 보편적으로 사용되는 형태 선택

2. **관련 용어 (Related Terms)** - {max(5, num_keywords // 3)}~{min(10, num_keywords // 2)}개:
   - 구체적이고 실체가 있는 고유명사 (제품명, 서비스명, 플랫폼명, 도구명)
   - 해당 분야의 주요 플레이어 (기업, 기관, 단체, 저명한 인물, 연구 그룹)
   - 구체적인 기술 사양, 버전, 모델명, 표준 규격 번호
   - 최근 1-2년 내 등장하거나 주목받는 신규 용어, 신조어, 트렌드
   - 경쟁 기술, 대체 솔루션, 유사 접근법의 구체적 명칭
   - 오픈소스 프로젝트명, 프레임워크, 라이브러리, API 명칭

3. **맥락 키워드 (Context Keywords)** - {max(5, num_keywords // 3)}~{min(10, num_keywords // 2)}개:
   - 주제가 속한 더 넓은 산업, 학문 분야, 사회적 영역
   - 관련 법규, 정책, 표준, 인증, 컴플라이언스 용어
   - 주제가 영향을 미치는 다른 분야나 산업 (교차점, 융합 영역)
   - 지정학적 맥락, 지역별 특성, 글로벌 vs 로컬 이슈
   - 시간적 맥락 (과거 배경, 현재 상황, 미래 전망과 관련된 용어)
   - 사회적 영향, 윤리적 고려사항, 지속가능성 관련 용어

**키워드 생성 상세 지침:**

1. **구체성과 검색가능성**:
   - 추상적이고 일반적인 용어는 피하고, 검색 결과를 좁힐 수 있는 구체적 용어 사용
   - "기술", "시스템", "솔루션" 같은 범용어보다는 정확한 명칭과 고유명사 선호
   - 가능한 경우 버전 번호, 연도, 세대 정보 포함 (예: 5G, Wi-Fi 6E, USB 4.0)

2. **시의성과 최신성**:
   - 2023-2024년에 등장했거나 주목받는 최신 용어 우선 포함
   - 해당 분야의 최근 주요 이벤트, 발표, 출시와 관련된 용어
   - 현재 진행 중인 논쟁, 이슈, 트렌드를 반영하는 키워드

3. **전문성과 깊이**:
   - 해당 분야 입문자가 아닌 전문가나 연구자가 사용할 법한 용어
   - 표면적 이해가 아닌 심층적 분석에 필요한 기술적 용어
   - 학술 데이터베이스, 특허 검색, 기술 포럼에서 사용되는 전문 용어

4. **다양성과 포괄성**:
   - 주제의 다양한 측면을 커버할 수 있도록 키워드 구성
   - 기술적, 비즈니스적, 사회적, 정책적 관점을 모두 포함
   - 긍정적 측면과 부정적 측면(위험, 한계, 비판) 모두 고려

5. **언어 선택 기준**:
   - 한국어/영어는 해당 용어가 실제로 더 많이 사용되는 형태로 선택
   - 국제 표준이나 고유명사는 원어 그대로 사용
   - 한국 특유의 맥락이나 용어가 있다면 포함"""

        if context:
            base_prompt += f"\n\n**추가 맥락 정보**: {context}"

        base_prompt += """

**품질 체크리스트**:
- 각 키워드가 독립적으로 의미 있는 검색 결과를 가져올 수 있는가?
- 단순 동의어나 번역이 아닌 고유한 가치를 가진 키워드인가?
- 실제 뉴스 헤드라인이나 논문 제목에 등장할 법한 용어인가?
- 2024년 현재 시점에서 여전히 유효하고 관련성 있는 용어인가?

**응답 형식 (반드시 유효한 JSON으로만 응답):**
{
    "primary_keywords": ["주제의 핵심을 나타내는 전문 용어들"],
    "related_terms": ["구체적인 제품/서비스/조직명 등 고유명사들"],
    "context_keywords": ["더 넓은 맥락과 관련 분야를 나타내는 용어들"],
    "confidence": 0.85
}

주의: 반드시 실재하고 검증 가능한 용어만 생성하세요. 추측이나 창작은 금지됩니다."""

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