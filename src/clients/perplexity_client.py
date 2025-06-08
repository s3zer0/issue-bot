"""
Perplexity AI API와의 통신을 전담하는 클라이언트 모듈.

이 모듈은 Perplexity API와 관련된 모든 HTTP 요청, 인증, 재시도 로직을 캡슐화하여
다른 비즈니스 로직 모듈들이 API의 세부 구현에 대해 알 필요가 없도록 합니다.
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional
from loguru import logger
from src.config import config  # 환경 설정 관리 모듈


class PerplexityClient:
    """Perplexity API 호출을 위한 비동기 클라이언트.

    Attributes:
        api_key (str): Perplexity API 키.
        base_url (str): API의 기본 URL.
        model (str): 사용할 LLM 모델 이름.
        timeout (int): HTTP 요청 타임아웃 시간(초).
        max_retries (int): 실패 시 최대 재시도 횟수.
        headers (Dict[str, str]): 모든 요청에 사용될 공통 HTTP 헤더.
    """

    def __init__(self, api_key: Optional[str] = None):
        """PerplexityClient 인스턴스를 초기화합니다.

        Args:
            api_key (Optional[str]): 사용할 API 키. None이면 config에서 가져옵니다.

        Raises:
            ValueError: API 키가 설정되지 않았을 경우.
        """
        # API 키 설정: 인자로 받은 키가 없으면 config에서 가져옴
        self.api_key = api_key or config.get_perplexity_api_key()
        if not self.api_key:
            raise ValueError("Perplexity API 키가 설정되지 않았습니다")

        # API 요청에 필요한 기본 설정
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"  # 사용할 LLM 모델
        self.timeout = 60  # HTTP 요청 타임아웃 (초)
        self.max_retries = 3  # 최대 재시도 횟수
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",  # 인증 헤더
            "Content-Type": "application/json"  # 요청 본문 형식
        }
        logger.info(f"PerplexityClient 초기화 완료 (모델: {self.model})")

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """API 호출 공통 로직.

        지수 백오프(exponential backoff)를 사용한 재시도 로직을 포함합니다.

        Args:
            prompt (str): LLM에 전달할 프롬프트.

        Returns:
            Dict[str, Any]: API로부터 받은 JSON 응답.

        Raises:
            ValueError: 여러 번의 재시도 후에도 API 호출에 실패했을 경우.
            httpx.HTTPStatusError: 429(Too Many Requests) 이외의 HTTP 오류 발생 시.
        """
        # 프롬프트의 일부를 로깅 (너무 길 경우 200자로 제한)
        logger.debug(f"Perplexity API 요청 프롬프트:\n---\n{prompt[:200]}...\n---")

        # API 요청 페이로드 구성
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise and objective information analysis expert. Always provide specific source URLs and publication dates when available."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10000,  # 최대 응답 토큰 수
            "temperature": 0.3  # 응답의 창의성 조절 (낮을수록 보수적)
        }

        # 비동기 HTTP 클라이언트 생성
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    # API POST 요청 실행
                    response = await client.post(self.base_url, headers=self.headers, json=payload)
                    response.raise_for_status()  # 2xx가 아닌 상태 코드에 대해 예외 발생
                    return response.json()  # 성공 시 JSON 응답 반환
                except httpx.HTTPStatusError as e:
                    # HTTP 상태 코드 에러 처리
                    logger.error(f"API HTTP Error (Status: {e.response.status_code}): {e.response.text}")
                    if e.response.status_code == 429:  # Rate Limit 오류 시
                        wait_time = 2 ** attempt  # 지수 백오프 시간 계산
                        logger.warning(f"Rate limit 초과. {wait_time}초 후 재시도... ({attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    raise  # 다른 HTTP 오류는 즉시 예외 발생
                except httpx.RequestError as e:
                    # 네트워크 요청 에러 처리
                    logger.error(f"API Request Error (Attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise ValueError("API 호출이 모든 재시도에 실패했습니다.")

        # 모든 재시도 실패 시 예외 발생
        raise ValueError("API 호출이 모든 재시도에 실패했습니다.")

    async def search_issues(self, keywords: List[str], time_period: str, max_results: int) -> Dict[str, Any]:
        """이슈 목록 검색을 위한 프롬프트 생성 및 API 호출.

        Args:
            keywords (List[str]): 검색할 키워드 목록.
            time_period (str): 검색 대상 기간 (예: '최근 1주일').
            max_results (int): 반환할 최대 결과 수.

        Returns:
            Dict[str, Any]: API로부터 받은 JSON 응답.
        """
        # 검색 요청을 위한 프롬프트 구성
        prompt = f"""'{", ".join(keywords)}' 키워드와 관련하여 '{time_period}' 동안 발행된 **개별 뉴스 기사, 블로그 포스트, 기술 문서**를 최대 {max_results}개 찾아줘.

        **필수 요구사항:**
        - 반드시 각 '개별 문서'의 내용을 기반으로 응답해야 해.
        - 실제 기사/문서의 웹 URL이나 도메인명을 반드시 포함해야 해.
        - 발행일은 YYYY-MM-DD 형식으로 명시해야 해.
        - 절대로 웹사이트의 메인 페이지나 기사 목록 페이지만 보고 요약해서는 안 돼.
        - 아래 형식을 반드시 준수해서, 각 기사 정보를 개별 항목으로 만들어줘.

        ## **[기사 제목]**
        **요약**: [기사 내용 요약 (구체적이고 상세하게)]
        **출처**: [실제 웹사이트명 또는 전체 URL (예: TechCrunch, https://techcrunch.com/2024/...)]
        **발행일**: [YYYY-MM-DD 형식]
        **카테고리**: [뉴스/블로그/기술문서/논문/보고서 중 하나]
        **기술적 핵심**: [기술적으로 중요한 핵심 포인트]
        **중요도**: [Critical/High/Medium/Low]
        **관련 키워드**: [검색 키워드 중 어떤 것과 관련있는지]
        
        주의: 출처가 명확하지 않거나 날짜를 모르는 경우, 그 항목은 포함하지 마세요."""

        # 공통 API 호출 메서드 실행
        return await self._make_api_call(prompt)

    async def collect_detailed_information(self, issue_title: str) -> Dict[str, Any]:
        """상세 정보 수집을 위한 프롬프트 생성 및 API 호출.

        Args:
            issue_title (str): 상세 정보를 수집할 이슈 제목.

        Returns:
            Dict[str, Any]: API로부터 받은 JSON 응답.
        """
        # 상세 정보 요청을 위한 프롬프트 구성
        prompt = f"""다음 이슈에 대해 **한국어로 상세하게 분석**해줘: **{issue_title}**.
        
        다음 정보를 반드시 포함해줘:
        1. 상세한 내용과 기술적 설명
        2. 해당 이슈가 발생하게 된 **배경 정보(Background Context)**
        3. 정보의 출처(웹사이트명, URL 등)가 있다면 명시
        4. 관련 날짜나 시기 정보가 있다면 포함
        5. 신뢰할 수 있는 구체적인 사실과 수치 위주로 작성"""

        # 요청 로깅 (제목의 일부만 기록)
        logger.info(f"상세 정보 요청: {issue_title[:50]}...")

        # 공통 API 호출 메서드 실행
        return await self._make_api_call(prompt)