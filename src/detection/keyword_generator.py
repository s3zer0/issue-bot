"""
키워드 생성 모듈 (기존 시스템과의 호환성 유지).

이 파일은 기존 코드와의 호환성을 위해 유지되며,
실제 구현은 keyword_generation 패키지로 이동했습니다.
"""

from typing import Optional
from loguru import logger

# 새로운 시스템으로 리다이렉트
from src.keyword_generation import (
    generate_keywords_for_topic as _generate_keywords,
    generate_multi_source_keywords,
    get_keyword_generation_status,
    configure_keyword_generation
)
from src.models import KeywordResult


# 기존 함수명 유지 (호환성)
async def generate_keywords_for_topic(
    topic: str,
    context: Optional[str] = None
) -> KeywordResult:
    """
    주제에 대한 키워드를 생성합니다 (멀티 소스).

    Args:
        topic: 키워드를 생성할 주제
        context: 추가 컨텍스트

    Returns:
        KeywordResult: 생성된 키워드 결과
    """
    logger.info(f"[호환성 래퍼] 키워드 생성 요청: {topic}")
    return await _generate_keywords(topic, context)


# 기존 클래스 (사용하지 않지만 호환성을 위해 유지)
class KeywordGenerator:
    """레거시 KeywordGenerator 클래스 (Deprecated)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        logger.warning(
            "KeywordGenerator 클래스는 deprecated되었습니다. "
            "대신 generate_keywords_for_topic() 함수를 사용하세요."
        )
        self.api_key = api_key
        self.model = model

    async def generate_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        num_keywords: int = 50
    ) -> KeywordResult:
        """레거시 메서드 (새로운 시스템으로 리다이렉트)."""
        return await generate_keywords_for_topic(topic, context)


# 편의 함수들
def create_keyword_generator(api_key: Optional[str] = None, model: Optional[str] = None):
    """레거시 팩토리 함수 (Deprecated)."""
    logger.warning(
        "create_keyword_generator()는 deprecated되었습니다. "
        "대신 generate_keywords_for_topic() 함수를 직접 사용하세요."
    )
    return KeywordGenerator(api_key, model)


# Export
__all__ = [
    'generate_keywords_for_topic',
    'generate_multi_source_keywords',
    'get_keyword_generation_status',
    'configure_keyword_generation',
    'KeywordGenerator',  # 호환성
    'create_keyword_generator'  # 호환성
]