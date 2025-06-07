# src/clients/perplexity_client.py
"""
Perplexity AI API 클라이언트 모듈
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional
from loguru import logger
from src.config import config


class PerplexityClient:
    """Perplexity API 통신을 전담하는 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.get_perplexity_api_key()
        if not self.api_key:
            raise ValueError("Perplexity API 키가 설정되지 않았습니다")

        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"
        self.timeout = 60
        self.max_retries = 3
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"PerplexityClient 초기화 완료 (모델: {self.model})")

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """API 호출 공통 로직"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise and objective information analysis expert."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10000,
            "temperature": 0.3
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(self.base_url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    logger.error(f"API HTTP Error (Status: {e.response.status_code}): {e.response.text}")
                    if e.response.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
                except httpx.RequestError as e:
                    logger.error(f"API Request Error (Attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
        raise ValueError("API call failed after multiple retries.")

    async def search_issues(self, keywords: List[str], time_period: str, max_results: int) -> Dict[str, Any]:
        """이슈 목록 검색을 위한 프롬프트 생성 및 API 호출"""
        prompt = f"""'{", ".join(keywords)}' 키워드와 관련하여 '{time_period}' 동안 발행된 **개별 뉴스 기사, 블로그 포스트, 기술 문서**를 최대 {max_results}개 찾아줘.

        **요구사항:**
        - 반드시 각 '개별 문서'의 내용을 기반으로 응답해야 해.
        - 절대로 웹사이트의 메인 페이지나 기사 목록 페이지만 보고 요약해서는 안 돼.
        - 아래 형식을 반드시 준수해서, 각 기사 정보를 개별 항목으로 만들어줘.

        ## **[기사 제목]**
        **요약**: [기사 내용 요약]
        **출처**: [출처 웹사이트 이름 또는 URL]
        **일자**: [발행 일자]
        **카테고리**: [뉴스, 블로그, 기술문서 등]"""
        return await self._make_api_call(prompt)

    async def collect_detailed_information(self, issue_title: str) -> Dict[str, Any]:
        """상세 정보 수집을 위한 프롬프트 생성 및 API 호출"""
        prompt = f"""다음 이슈에 대해 **한국어로 상세하게 분석**해줘: **{issue_title}**.
        상세 내용과 함께, 해당 이슈가 발생하게 된 **배경 정보(Background Context)**도 찾아서 포함해줘."""
        logger.info(f"Requesting detailed info for: {issue_title[:50]}...")
        return await self._make_api_call(prompt)