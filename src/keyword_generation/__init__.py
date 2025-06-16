# src/keyword_generation/__init__.py
"""
멀티 소스 키워드 생성 시스템.

기존 시스템과의 호환성을 유지하면서 새로운 멀티 소스 기능을 제공합니다.
"""

from typing import Optional, List
from loguru import logger

from src.config import config
from src.models import KeywordResult

from .base import BaseKeywordExtractor, KeywordItem, KeywordImportance
from .manager import MultiSourceKeywordManager, MultiSourceKeywordResult
from .similarity import KeywordSimilarityAnalyzer
from .extractors import (
    GPTKeywordExtractor,
    GrokKeywordExtractor,
    PerplexityKeywordExtractor
)


# 전역 매니저 인스턴스
_global_manager: Optional[MultiSourceKeywordManager] = None


def get_keyword_manager() -> MultiSourceKeywordManager:
    """전역 키워드 매니저 인스턴스를 반환합니다."""
    global _global_manager

    if _global_manager is None:
        _global_manager = _initialize_default_manager()

    return _global_manager


def _initialize_default_manager() -> MultiSourceKeywordManager:
    """기본 설정으로 키워드 매니저를 초기화합니다."""
    logger.info("멀티 소스 키워드 매니저 초기화 중...")

    # 유사도 분석기 생성
    similarity_analyzer = KeywordSimilarityAnalyzer(similarity_threshold=0.85)

    # 추출기 리스트 생성
    extractors = []

    # 1. GPT 추출기 (OpenAI API 키가 있는 경우)
    if config.get_openai_api_key():
        try:
            gpt_extractor = GPTKeywordExtractor()
            extractors.append(gpt_extractor)
            logger.success("GPT 키워드 추출기 활성화")
        except Exception as e:
            logger.warning(f"GPT 추출기 초기화 실패: {e}")

    # 2. Perplexity 추출기 (Perplexity API 키가 있는 경우)
    if config.get_perplexity_api_key():
        try:
            perplexity_extractor = PerplexityKeywordExtractor()
            extractors.append(perplexity_extractor)
            logger.success("Perplexity 키워드 추출기 활성화")
        except Exception as e:
            logger.warning(f"Perplexity 추출기 초기화 실패: {e}")

    # 3. Grok 추출기 (실제 API 키 확인 후 동작)
    grok_api_key = config.get_grok_api_key()
    if grok_api_key:
        try:
            grok_extractor = GrokKeywordExtractor(api_key=grok_api_key)
            extractors.append(grok_extractor)
            logger.success("Grok 키워드 추출기 활성화 (실제 API 모드)")
        except Exception as e:
            logger.warning(f"Grok 추출기 초기화 실패: {e}")
    else:
        # API 키가 없는 경우에만 시뮬레이션 모드로 동작
        try:
            grok_extractor = GrokKeywordExtractor()  # API 키 없이 시뮬레이션 모드
            extractors.append(grok_extractor)
            logger.success("Grok 키워드 추출기 활성화 (시뮬레이션 모드)")
        except Exception as e:
            logger.warning(f"Grok 추출기 초기화 실패: {e}")

    # 매니저 생성
    manager = MultiSourceKeywordManager(
        extractors=extractors,
        similarity_analyzer=similarity_analyzer
    )

    logger.info(f"총 {len(extractors)}개의 키워드 추출기가 활성화되었습니다")
    return manager


# 기존 인터페이스와의 호환성을 위한 래퍼 함수
async def generate_keywords_for_topic(
    topic: str,
    context: Optional[str] = None
) -> KeywordResult:
    """
    기존 시스템과 호환되는 키워드 생성 함수.

    Args:
        topic: 키워드를 생성할 주제
        context: 추가 컨텍스트

    Returns:
        KeywordResult: 기존 형식의 키워드 결과
    """
    # 입력 검증: 빈 토픽 방지
    if not topic or not topic.strip():
        logger.warning("빈 토픽이 제공됨. 기본 키워드로 폴백.")
        return _create_fallback_result("technology trends")
    
    # 토픽 정규화
    topic = topic.strip()
    logger.info(f"키워드 생성 요청: '{topic}'")

    # 멀티 소스 매니저 사용
    manager = get_keyword_manager()

    # 추출기가 없는 경우 폴백
    if not manager.extractors:
        logger.error("활성화된 키워드 추출기가 없습니다")
        return _create_fallback_result(topic)

    try:
        # 멀티 소스 키워드 생성
        multi_result = await manager.generate_keywords(topic, context)

        # 기존 형식으로 변환
        legacy_result = multi_result.to_legacy_format(topic)

        # 결과 검증: 빈 키워드 방지
        if not legacy_result.primary_keywords and not legacy_result.related_terms:
            logger.warning("생성된 키워드가 모두 비어있음. 폴백 사용.")
            return _create_fallback_result(topic)
        
        # 빈 키워드 필터링
        legacy_result.primary_keywords = [kw for kw in legacy_result.primary_keywords if kw and kw.strip()]
        legacy_result.related_terms = [kw for kw in legacy_result.related_terms if kw and kw.strip()]
        legacy_result.context_keywords = [kw for kw in legacy_result.context_keywords if kw and kw.strip()]

        # 통계 로그
        logger.info(
            f"키워드 생성 완료: "
            f"총 {len(multi_result.keywords)}개 키워드 "
            f"(높음: {multi_result.high_importance_count}, "
            f"보통: {multi_result.normal_importance_count}, "
            f"낮음: {multi_result.low_importance_count})"
        )

        return legacy_result

    except Exception as e:
        logger.error(f"키워드 생성 중 오류 발생: {e}")
        return _create_fallback_result(topic)


def _create_fallback_result(topic: str) -> KeywordResult:
    """오류 발생 시 폴백 결과 생성."""
    return KeywordResult(
        topic=topic,
        primary_keywords=[topic],
        related_terms=[],
        context_keywords=[],
        confidence_score=0.1,
        generation_time=0.0,
        raw_response="Error occurred during keyword generation"
    )


# 고급 사용을 위한 직접 접근 함수
async def generate_multi_source_keywords(
    topic: str,
    context: Optional[str] = None,
    extractors: Optional[List[BaseKeywordExtractor]] = None
) -> MultiSourceKeywordResult:
    """
    멀티 소스 키워드 생성 (고급 사용).

    Args:
        topic: 키워드를 생성할 주제
        context: 추가 컨텍스트
        extractors: 사용할 추출기 리스트 (None이면 기본값 사용)

    Returns:
        MultiSourceKeywordResult: 멀티 소스 키워드 결과
    """
    if extractors:
        # 커스텀 추출기로 임시 매니저 생성
        manager = MultiSourceKeywordManager(
            extractors=extractors,
            similarity_analyzer=KeywordSimilarityAnalyzer()
        )
    else:
        # 전역 매니저 사용
        manager = get_keyword_manager()

    return await manager.generate_keywords(topic, context)


# 설정 함수
def configure_keyword_generation(
    enable_gpt: bool = True,
    enable_perplexity: bool = True,
    enable_grok: bool = True,
    similarity_threshold: float = 0.85
):
    """
    키워드 생성 시스템을 구성합니다.

    Args:
        enable_gpt: GPT 추출기 활성화 여부
        enable_perplexity: Perplexity 추출기 활성화 여부
        enable_grok: Grok 추출기 활성화 여부
        similarity_threshold: 유사도 임계값
    """
    global _global_manager

    extractors = []

    if enable_gpt and config.get_openai_api_key():
        extractors.append(GPTKeywordExtractor())

    if enable_perplexity and config.get_perplexity_api_key():
        extractors.append(PerplexityKeywordExtractor())

    if enable_grok:
        grok_api_key = config.get_grok_api_key()
        if grok_api_key:
            # 실제 API 키가 있으면 실제 모드로 동작
            extractors.append(GrokKeywordExtractor(api_key=grok_api_key))
            logger.info("Grok 키워드 추출기 활성화됨 (실제 API 모드)")
        else:
            # API 키가 없으면 시뮬레이션 모드로 동작
            extractors.append(GrokKeywordExtractor())
            logger.info("Grok 키워드 추출기 활성화됨 (시뮬레이션 모드)")

    _global_manager = MultiSourceKeywordManager(
        extractors=extractors,
        similarity_analyzer=KeywordSimilarityAnalyzer(similarity_threshold)
    )

    logger.info(f"키워드 생성 시스템 재구성 완료 (추출기: {len(extractors)}개)")



# 상태 확인 함수
def get_keyword_generation_status() -> dict:
    """키워드 생성 시스템의 상태를 반환합니다."""
    manager = get_keyword_manager()

    # 각 추출기의 상태 확인
    extractor_status = {}
    for extractor in manager.extractors:
        if hasattr(extractor, 'simulation_mode'):
            mode = "시뮬레이션" if extractor.simulation_mode else "실제 API"
            extractor_status[extractor.name] = mode
        else:
            extractor_status[extractor.name] = "실제 API"

    return {
        'active_extractors': [e.name for e in manager.extractors],
        'extractor_modes': extractor_status,
        'total_extractors': len(manager.extractors),
        'similarity_threshold': manager.similarity_analyzer.similarity_threshold,
        'available_apis': {
            'gpt': bool(config.get_openai_api_key()),
            'perplexity': bool(config.get_perplexity_api_key()),
            'grok': bool(config.get_grok_api_key())
        }
    }

# Export
__all__ = [
    # 기존 호환성
    'generate_keywords_for_topic',
    # 멀티 소스 기능
    'generate_multi_source_keywords',
    'get_keyword_manager',
    'configure_keyword_generation',
    'get_keyword_generation_status',
    # 클래스들
    'BaseKeywordExtractor',
    'MultiSourceKeywordManager',
    'MultiSourceKeywordResult',
    'KeywordItem',
    'KeywordImportance',
    # 추출기들
    'GPTKeywordExtractor',
    'GrokKeywordExtractor',
    'PerplexityKeywordExtractor'
]
