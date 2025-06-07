# tests/test_config.py
"""
config.py 모듈 단위 테스트 (최종 완성본)
"""
import pytest
import os
from unittest.mock import patch

# 리팩토링된 구조에 맞춰 import 경로 수정
from src.config import Config

@patch.dict(os.environ, {}, clear=True)
class TestConfig:
    """Config 클래스의 다양한 시나리오를 테스트합니다."""

    # ... (test_get_keys_with_valid_values, test_get_keys_return_none_for_placeholder, test_creates_sample_when_no_env_file, test_is_development_mode 는 기존과 동일)
    def test_get_keys_with_valid_values(self):
        """환경 변수가 올바르게 설정되었을 때 키를 잘 반환하는지 테스트"""
        with patch.dict(os.environ, {
            "DISCORD_BOT_TOKEN": "real_discord_token",
            "OPENAI_API_KEY": "real_openai_key",
        }):
            with patch('src.config.load_dotenv', return_value=True):
                cfg = Config()
                assert cfg.get_discord_token() == "real_discord_token"
                assert cfg.get_openai_api_key() == "real_openai_key"

    def test_get_keys_return_none_for_placeholder(self):
        """API 키가 플레이스홀더 값일 때 None을 반환하는지 테스트"""
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "your_token_here"}):
            with patch('src.config.load_dotenv', return_value=True):
                cfg = Config()
                assert cfg.get_discord_token() is None

    @patch('src.config.load_dotenv', return_value=False)
    @patch('src.config.Path.exists', return_value=False)
    @patch('builtins.open')
    def test_creates_sample_when_no_env_file(self, mock_open, mock_path_exists, mock_load_dotenv):
        """ .env 파일이 없을 때 샘플 파일을 생성하는지 테스트"""
        Config()
        mock_open.assert_called_once()
        assert '.env.example' in str(mock_open.call_args[0][0])

    @pytest.mark.parametrize("env_value,expected", [
        ('true', True), ('True', True), ('false', False), ('', False), (None, False)
    ])
    def test_is_development_mode(self, env_value, expected):
        """is_development_mode가 환경 변수 값을 올바르게 파싱하는지 테스트"""
        env_dict = {"DEVELOPMENT_MODE": env_value} if env_value is not None else {}
        with patch.dict(os.environ, env_dict, clear=True):
             with patch('src.config.load_dotenv', return_value=True):
                cfg = Config()
                assert cfg.is_development_mode() is expected

    # 단계 확인 테스트 로직
    @pytest.mark.parametrize("s1,s2,s3,expected_stage", [
        (False, False, False, 0), # 아무것도 없을 때 0단계
        (True, False, False, 1),  # Discord 토큰만 있을 때 1단계
        (True, True, False, 2),   # OpenAI 키까지 있을 때 2단계
        (True, True, True, 4),    # Perplexity 키까지 모두 있을 때 4단계
    ])
    def test_get_current_stage(self, s1, s2, s3, expected_stage):
        """모든 단계별 조합에 대해 정확한 현재 단계를 반환하는지 테스트"""
        # Config 객체 내부의 메서드를 mock
        with patch.object(Config, 'validate_stage1_requirements', return_value=s1), \
             patch.object(Config, 'validate_stage2_requirements', return_value=s2), \
             patch.object(Config, 'validate_stage3_requirements', return_value=s3):

            with patch('src.config.load_dotenv', return_value=True):
                cfg = Config()
                assert cfg.get_current_stage() == expected_stage