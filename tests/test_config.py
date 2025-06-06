"""
config.py 모듈 단위 테스트
"""
import pytest
import os
from unittest.mock import patch
from pathlib import Path

# 테스트 대상 모듈 임포트
from src.config import Config

@patch.dict(os.environ, {}, clear=True)
class TestConfigDefaults:
    """환경 변수가 설정되지 않았을 때의 동작을 테스트합니다."""

    @patch('src.config.load_dotenv', return_value=False) # .env 파일이 없는 것처럼 시뮬레이션
    @patch('builtins.open')
    def test_create_sample_env_file_if_not_exists(self, mock_open, mock_load_dotenv):
        """ .env 파일이 없을 때 .env.example 파일을 생성하는지 테스트 """

        # 💡 [수정] Path.exists에 대한 mock을 제거하여 실제 경로 계산 로직이 동작하도록 함
        # 대신 open 함수가 올바른 경로로 호출되었는지만 검증
        with patch('src.config.Path.exists', return_value=False):
            Config()

        # open 함수가 호출되었는지 확인
        mock_open.assert_called_once()
        # open 함수에 전달된 첫 번째 인자(파일 경로)를 가져옴
        call_args = mock_open.call_args[0]
        called_path = call_args[0]

        # 경로가 Path 객체이고, 이름이 '.env.example'로 끝나는지 검증
        assert isinstance(called_path, Path)
        assert called_path.name == '.env.example'


    def test_get_openai_settings_with_defaults(self):
        """ OpenAI 관련 설정들이 기본값을 잘 반환하는지 테스트 """
        with patch('src.config.load_dotenv', return_value=False):
            cfg = Config()
            assert cfg.get_openai_temperature() == 0.7
            assert cfg.get_openai_max_tokens() == 1500
            assert cfg.get_max_retry_count() == 3

@patch.dict(os.environ, {}, clear=True)
class TestStageCalculation:
    """get_current_stage 함수의 정확성을 테스트합니다."""

    @pytest.mark.parametrize("s1,s2,s3,expected_stage", [
        (False, False, False, 0),
        (True, False, False, 1),
        (True, True, False, 2),
        (True, True, True, 4),
    ])
    def test_get_current_stage(self, s1, s2, s3, expected_stage):
        """ 모든 단계별 조합에 대해 정확한 현재 단계를 반환하는지 테스트 """
        with patch('src.config.Config.validate_stage1_requirements', return_value=s1), \
             patch('src.config.Config.validate_stage2_requirements', return_value=s2), \
             patch('src.config.Config.validate_stage3_requirements', return_value=s3):

            with patch('src.config.load_dotenv', return_value=True):
                cfg = Config()
                assert cfg.get_current_stage() == expected_stage