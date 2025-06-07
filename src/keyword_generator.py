"""
키워드 생성 모듈
- LLM(OpenAI)을 활용하여 주제 기반 키워드를 자동 생성합니다.
- 생성된 키워드는 이슈 모니터링, 검색 최적화, 분석에 사용됩니다.
"""

import asyncio
import json
import re
import time
from typing import List, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    from openai import AsyncOpenAI, AuthenticationError
except ImportError:
    logger.error("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행해주세요.")
    class AsyncOpenAI: pass  # OpenAI 라이브러리 미설치 시 더미 클래스 정의
    class AuthenticationError(Exception): pass  # 더미 예외 클래스 정의

from src.config import config  # 환경 설정 로드
from src.models import KeywordResult  # 중앙 데이터 모델 import


@dataclass
class KeywordResult:
    """키워드 생성 결과를 구조화된 형태로 저장하는 데이터 클래스.

    Attributes:
        topic (str): 키워드 생성의 기반이 되는 주제.
        primary_keywords (List[str]): 주제의 본질을 포착하는 핵심 키워드.
        related_terms (List[str]): 고유명사, 제품명, 최신 용어 등 관련 용어.
        context_keywords (List[str]): 주제의 산업, 정책, 사회적 맥락 관련 키워드.
        confidence_score (float): 생성된 키워드의 신뢰도 점수 (0.0 ~ 1.0).
        generation_time (float): 키워드 생성에 소요된 시간 (초).
        raw_response (str): LLM의 원본 응답 데이터.
    """
    topic: str
    primary_keywords: List[str]
    related_terms: List[str]
    context_keywords: List[str]
    confidence_score: float
    generation_time: float
    raw_response: str


class KeywordGenerator:
    """OpenAI LLM을 활용한 키워드 생성 클래스.

    주제와 맥락을 기반으로 전문적이고 검색에 최적화된 키워드를 생성합니다.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """KeywordGenerator 인스턴스를 초기화합니다.

        Args:
            api_key (Optional[str]): OpenAI API 키. None일 경우 환경 변수에서 로드.
            model (Optional[str]): 사용할 LLM 모델. None일 경우 기본 모델 사용.

        Raises:
            ValueError: API 키가 제공되지 않거나 유효하지 않을 경우.
        """
        self.api_key = api_key or config.get_openai_api_key()
        self.model = model or config.get_openai_model()

        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.max_retries = config.get_max_retry_count()  # 최대 재시도 횟수
        self.timeout = config.get_keyword_generation_timeout()  # 요청 타임아웃
        self.temperature = config.get_openai_temperature()  # LLM 응답 다양성 조절
        self.max_tokens = config.get_openai_max_tokens()  # 최대 토큰 수
        logger.info(f"KeywordGenerator 초기화 완료 (모델: {self.model})")

    async def generate_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        num_keywords: int = 50
    ) -> KeywordResult:
        """주제와 선택적 맥락을 기반으로 키워드를 비동기적으로 생성합니다.

        Args:
            topic (str): 키워드 생성의 기반 주제.
            context (Optional[str]): 주제에 대한 추가 맥락 정보.
            num_keywords (int): 생성할 키워드의 목표 개수. 기본값은 50.

        Returns:
            KeywordResult: 생성된 키워드와 메타데이터를 포함한 결과 객체.

        Raises:
            ValueError: 주제가 비어 있거나 유효하지 않을 경우.
            Exception: LLM 호출 또는 응답 파싱 중 오류 발생 시.
        """
        start_time = time.time()
        logger.info(f"키워드 생성 시작: '{topic}' (모델: {self.model})")

        if not topic or not topic.strip():
            raise ValueError("주제가 비어있습니다")

        topic = topic.strip()

        try:
            # 1. LLM에 보낼 프롬프트 생성
            prompt = self._build_prompt(topic, context, num_keywords)

            # 2. LLM 호출로 원본 응답 수집
            raw_response = await self._call_llm(prompt)

            # 3. 응답 파싱 및 결과 객체 생성
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
        """LLM에 전달할 키워드 생성 프롬프트를 구성합니다.

        전문적이고 검색에 최적화된 키워드를 생성하도록 상세한 지침을 포함합니다.

        Args:
            topic (str): 키워드 생성의 기반 주제.
            context (Optional[str]): 추가 맥락 정보.
            num_keywords (int): 생성할 키워드의 목표 개수.

        Returns:
            str: 구성된 프롬프트 문자열.
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
   - 한국 특유의 맥락이나 용어가 있다면 포함
"""

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
        """OpenAI LLM API를 호출하여 키워드를 생성합니다.

        GPT-4o 모델에 최적화된 파라미터를 적용하며, 재시도 로직을 포함합니다.

        Args:
            prompt (str): LLM에 전달할 프롬프트.

        Returns:
            str: LLM의 원본 응답 문자열.

        Raises:
            ValueError: API 인증 실패, 사용량 초과, 크레딧 부족 등으로 최종 실패 시.
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM API 호출 시도 {attempt + 1}/{self.max_retries}")

                # 기본 요청 파라미터 설정
                request_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 특정 주제에 대한 깊이 있는 분석을 위해 검색 키워드를 생성하는 IT 전문 분석가입니다. 반드시 유효한 JSON 형식으로만 응답해야 합니다."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout
                }

                # GPT-4o 모델일 경우 추가 최적화 파라미터 적용
                if self.model == "gpt-4o":
                    request_params.update({
                        "frequency_penalty": 0.3,  # 반복된 단어 사용 억제
                        "presence_penalty": 0.3,  # 새로운 주제 탐색 장려
                        "response_format": {"type": "json_object"}  # JSON 출력 강제
                    })
                    logger.debug("GPT-4o 최적화 파라미터 적용")

                # API 호출 및 응답 처리
                response = await self.client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM 응답이 비어있습니다")
                return content.strip()

            except AuthenticationError as e:
                logger.error(f"OpenAI 인증 오류: {e}")
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
                await asyncio.sleep(2 ** attempt)  # 지수 백오프 재시도

    def _parse_response(self, topic: str, raw_response: str, generation_time: float) -> KeywordResult:
        """LLM 응답을 파싱하여 KeywordResult 객체로 변환합니다.

        Args:
            topic (str): 키워드 생성의 기반 주제.
            raw_response (str): LLM의 원본 응답.
            generation_time (float): 키워드 생성에 소요된 시간.

        Returns:
            KeywordResult: 파싱된 키워드 결과 객체.

        Raises:
            ValueError: 응답에 유효한 JSON이 없거나 필수 필드가 누락된 경우.
        """
        try:
            # JSON 객체 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            if not json_match:
                raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")

            data = json.loads(json_match.group())

            # 필수 필드 확인
            required_fields = ['primary_keywords', 'related_terms', 'context_keywords']
            if not all(field in data for field in required_fields):
                raise ValueError(f"필수 필드가 누락되었습니다: {required_fields}")

            # 키워드 정제 및 중복 제거
            primary_keywords = self._clean_keywords(data.get('primary_keywords', []))
            related_terms = self._clean_keywords(data.get('related_terms', []))
            context_keywords = self._clean_keywords(data.get('context_keywords', []))
            confidence_score = min(1.0, max(0.0, float(data.get('confidence', 0.8))))

            # 핵심 키워드가 없으면 주제를 기본 키워드로 사용
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
        """키워드 리스트를 정제하고 중복을 제거합니다.

        Args:
            keywords (List[Any]): 정제할 원본 키워드 리스트.

        Returns:
            List[str]: 정제된 고유 키워드 리스트 (최대 12개).
        """
        if not isinstance(keywords, list):
            logger.warning(f"키워드 데이터가 리스트가 아닙니다: {type(keywords)}")
            return []

        cleaned = []
        for keyword in keywords:
            if isinstance(keyword, str):
                keyword = keyword.strip().strip('"\'')
                if len(keyword) > 1:  # 너무 짧은 키워드 제외
                    cleaned.append(keyword)

        # 대소문자 구분 없이 중복 제거
        seen = set()
        unique_keywords = []
        for keyword in cleaned:
            lower_keyword = keyword.lower()
            if lower_keyword not in seen:
                seen.add(lower_keyword)
                unique_keywords.append(keyword)
        return unique_keywords[:12]  # 최대 12개로 제한

    def _create_fallback_result(self, topic: str, raw_response: str, generation_time: float) -> KeywordResult:
        """응답 파싱 실패 시 기본 결과를 생성합니다.

        Args:
            topic (str): 키워드 생성의 기반 주제.
            raw_response (str): LLM의 원본 응답.
            generation_time (float): 키워드 생성에 소요된 시간.

        Returns:
            KeywordResult: 기본 키워드와 낮은 신뢰도를 포함한 결과 객체.
        """
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
        """모든 키워드 카테고리를 단일 리스트로 결합하여 반환합니다.

        Args:
            result (KeywordResult): 키워드 결과 객체.

        Returns:
            List[str]: 중복 제거된 전체 키워드 리스트.
        """
        all_keywords = (result.primary_keywords + result.related_terms + result.context_keywords)
        return list(dict.fromkeys(all_keywords))

    def format_keywords_summary(self, result: KeywordResult) -> str:
        """키워드 결과를 읽기 쉬운 요약 문자열로 포맷팅합니다.

        Args:
            result (KeywordResult): 요약할 키워드 결과 객체.

        Returns:
            str: 포맷팅된 요약 문자열.
        """
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
    """KeywordGenerator 인스턴스를 생성하는 팩토리 함수.

    Args:
        api_key (Optional[str]): OpenAI API 키. None일 경우 환경 변수에서 로드.
        model (Optional[str]): 사용할 LLM 모델. None일 경우 기본 모델 사용.

    Returns:
        KeywordGenerator: 초기화된 KeywordGenerator 인스턴스.
    """
    return KeywordGenerator(api_key=api_key, model=model)


async def generate_keywords_for_topic(topic: str, context: Optional[str] = None) -> KeywordResult:
    """주제에 대한 키워드를 생성하는 고수준 래퍼 함수.

    Args:
        topic (str): 키워드 생성의 기반 주제.
        context (Optional[str]): 추가 맥락 정보.

    Returns:
        KeywordResult: 생성된 키워드 결과 객체.
    """
    generator = create_keyword_generator()
    return await generator.generate_keywords(topic, context)


if __name__ == "__main__":
    print("🧪 키워드 생성기 테스트")
    print("pytest로 테스트를 실행하세요:")
    print("pytest tests/test_keyword_generator.py -v")