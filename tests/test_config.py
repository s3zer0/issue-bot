"""
pytest 공통 설정 및 픽스처
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# src 모듈 경로 추가 (모든 테스트에서 공통으로 사용)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)


@pytest.fixture
def mock_config():
    """Mock Config 객체 픽스처"""
    with patch.dict(os.environ, {
        'DISCORD_BOT_TOKEN': 'test_discord_token',
        'OPENAI_API_KEY': 'test_openai_key',
        'PERPLEXITY_API_KEY': 'test_perplexity_key',
        'DEBUG': 'True'
    }):
        from src.config import Config
        yield Config()


@pytest.fixture
def mock_discord_interaction():
    """Mock Discord Interaction 객체 픽스처"""
    interaction = MagicMock()
    interaction.response.defer = MagicMock()
    interaction.followup.send = MagicMock()
    return interaction


@pytest.fixture
def sample_topics():
    """테스트용 샘플 주제들"""
    return [
        "AI 기술 발전",
        "암호화폐 시장 동향",
        "기후변화 대응정책",
        "전기차 산업",
        "우주 탐사",
        "양자컴퓨팅"
    ]


@pytest.fixture
def sample_periods():
    """테스트용 샘플 기간들"""
    return [
        "1일", "3일", "7일",
        "1주일", "2주일", "4주일",
        "1개월", "3개월", "6개월",
        "12시간", "24시간", "72시간"
    ]


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """테스트 환경 전역 설정"""
    # 테스트 실행 전 로깅 레벨 조정
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    # 테스트 시작 알림
    print("\n🧪 pytest 테스트 시작")

    yield

    # 테스트 완료 알림
    print("✅ pytest 테스트 완료\n")


@pytest.fixture
def mock_api_responses():
    """Mock API 응답 데이터"""
    return {
        'openai': {
            'choices': [{
                'message': {
                    'content': '테스트 키워드1, 테스트 키워드2, 테스트 키워드3'
                }
            }]
        },
        'perplexity': {
            'choices': [{
                'message': {
                    'content': '테스트 이슈 내용입니다.'
                }
            }]
        }
    }


# 커스텀 마커 설정
def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line("markers", "unit: 단위 테스트")
    config.addinivalue_line("markers", "integration: 통합 테스트")
    config.addinivalue_line("markers", "slow: 느린 테스트")
    config.addinivalue_line("markers", "api: API 호출이 필요한 테스트")


def pytest_collection_modifyitems(config, items):
    """테스트 수집 후 수정"""
    # slow 마커가 있는 테스트는 마지막에 실행
    slow_tests = []
    regular_tests = []

    for item in items:
        if "slow" in item.keywords:
            slow_tests.append(item)
        else:
            regular_tests.append(item)

    items[:] = regular_tests + slow_tests


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API 응답 픽스처"""
    return {
        'choices': [{
            'message': {
                'content': '''
                {
                    "primary_keywords": ["AI", "인공지능", "머신러닝"],
                    "related_terms": ["딥러닝", "신경망", "알고리즘"],
                    "synonyms": ["Artificial Intelligence", "기계학습"],
                    "context_keywords": ["기술혁신", "자동화", "데이터분석"],
                    "confidence": 0.9
                }
                '''
            }
        }]
    }

@pytest.fixture
def sample_keyword_result():
    """테스트용 KeywordResult 픽스처"""
    from src.keyword_generator import KeywordResult

    return KeywordResult(
        topic="AI 기술 발전",
        primary_keywords=["AI", "인공지능", "머신러닝", "딥러닝"],
        related_terms=["신경망", "알고리즘", "빅데이터"],
        synonyms=["Artificial Intelligence", "기계학습"],
        context_keywords=["기술혁신", "자동화", "디지털트랜스포메이션"],
        confidence_score=0.85,
        generation_time=2.5,
        raw_response="mock response"
    )


@pytest.fixture
def mock_keyword_generator():
    """Mock KeywordGenerator 픽스처"""
    from unittest.mock import patch
    from src.keyword_generator import create_keyword_generator

    with patch('keyword_generator.AsyncOpenAI'):
        generator = create_keyword_generator(api_key="test_key")
        return generator


@pytest.fixture
def valid_openai_env():
    """유효한 OpenAI 환경변수 설정 픽스처"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key_12345'
    }):
        yield


@pytest.fixture
def mock_config():
    """Mock Config 객체 픽스처 - 업데이트"""
    with patch.dict(os.environ, {
        'DISCORD_BOT_TOKEN': 'test_discord_token',
        'OPENAI_API_KEY': 'test_openai_key',
        'PERPLEXITY_API_KEY': 'test_perplexity_key',
        'DEVELOPMENT_MODE': 'True'
    }):
        from src.config import Config
        yield Config()


# pytest 마커 등록
def pytest_configure(config):
    """pytest 설정 - 마커 등록"""
    config.addinivalue_line("markers", "unit: 단위 테스트")
    config.addinivalue_line("markers", "integration: 통합 테스트")
    config.addinivalue_line("markers", "slow: 느린 테스트")
    config.addinivalue_line("markers", "api: API 호출이 필요한 테스트")
    config.addinivalue_line("markers", "asyncio: 비동기 테스트")