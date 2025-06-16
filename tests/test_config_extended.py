"""
Extended tests for config module to achieve higher coverage.

These tests cover the missing functionality from the original test_config.py
including error handling, file operations, and all getter methods.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from src.config import Config


class TestConfigExtended:
    """Extended tests for Config class to achieve better coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('src.config.logger')
    def test_setup_environment_exception_handling(self, mock_logger):
        """Test _setup_environment with exception handling."""
        # Create a config instance first
        config = Config()
        
        # Clear any previous calls to logger
        mock_logger.warning.reset_mock()
        
        # Now test the _setup_environment method directly with an exception
        with patch('pathlib.Path.__new__', side_effect=Exception("Path error")):
            config._setup_environment()
            # Verify that the warning was logged
            mock_logger.warning.assert_called_once()
            assert "환경 설정 중 오류 발생" in str(mock_logger.warning.call_args)
    
    @patch('src.config.load_dotenv')
    @patch('src.config.logger')
    def test_load_env_file_missing_creates_sample(self, mock_logger, mock_load_dotenv):
        """Test that missing .env file triggers sample creation."""
        mock_load_dotenv.return_value = False
        
        with patch.object(self.config, '_create_sample_env_file') as mock_create:
            self.config._load_env_file()
            mock_create.assert_called_once()
            mock_logger.warning.assert_called()
    
    def test_create_sample_env_file_success(self):
        """Test successful creation of sample .env file."""
        test_path = Path(self.test_dir)
        
        with patch('src.config.logger') as mock_logger:
            self.config._create_sample_env_file(test_path)
            
            sample_file = test_path / '.env.example'
            assert sample_file.exists()
            
            # Check content
            content = sample_file.read_text(encoding='utf-8')
            assert 'DISCORD_BOT_TOKEN' in content
            assert 'OPENAI_API_KEY' in content
            assert 'PERPLEXITY_API_KEY' in content
            
            mock_logger.info.assert_called()
    
    def test_create_sample_env_file_already_exists(self):
        """Test that existing sample file is not overwritten."""
        test_path = Path(self.test_dir)
        sample_file = test_path / '.env.example'
        
        # Create existing file
        sample_file.write_text("existing content")
        original_content = sample_file.read_text()
        
        self.config._create_sample_env_file(test_path)
        
        # Content should not change
        assert sample_file.read_text() == original_content
    
    @patch('builtins.open', side_effect=IOError("Write permission denied"))
    @patch('src.config.logger')
    def test_create_sample_env_file_io_error(self, mock_logger, mock_open_error):
        """Test handling of IO error during sample file creation."""
        test_path = Path(self.test_dir)
        
        self.config._create_sample_env_file(test_path)
        mock_logger.error.assert_called()
    
    def test_get_discord_token_valid(self):
        """Test getting valid Discord token."""
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'valid_token_123'}):
            token = self.config.get_discord_token()
            assert token == 'valid_token_123'
    
    def test_get_discord_token_placeholder(self):
        """Test that placeholder token returns None."""
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'your_token_here'}):
            token = self.config.get_discord_token()
            assert token is None
    
    def test_get_discord_token_missing(self):
        """Test that missing token returns None."""
        with patch.dict(os.environ, {}, clear=True):
            token = self.config.get_discord_token()
            assert token is None
    
    def test_get_openai_api_key_valid(self):
        """Test getting valid OpenAI API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-valid_key_123'}):
            key = self.config.get_openai_api_key()
            assert key == 'sk-valid_key_123'
    
    def test_get_openai_api_key_placeholder(self):
        """Test that placeholder OpenAI key returns None."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'your_key_here'}):
            key = self.config.get_openai_api_key()
            assert key is None
    
    def test_get_openai_api_key_missing(self):
        """Test that missing OpenAI key returns None."""
        with patch.dict(os.environ, {}, clear=True):
            key = self.config.get_openai_api_key()
            assert key is None
    
    def test_get_perplexity_api_key_valid(self):
        """Test getting valid Perplexity API key."""
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'pplx-valid_key_123'}):
            key = self.config.get_perplexity_api_key()
            assert key == 'pplx-valid_key_123'
    
    def test_get_perplexity_api_key_placeholder(self):
        """Test that placeholder Perplexity key returns None."""
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'your_key_here'}):
            key = self.config.get_perplexity_api_key()
            assert key is None
    
    def test_get_perplexity_api_key_missing(self):
        """Test that missing Perplexity key returns None."""
        with patch.dict(os.environ, {}, clear=True):
            key = self.config.get_perplexity_api_key()
            assert key is None
    
    def test_get_grok_api_key_valid(self):
        """Test getting valid Grok API key."""
        with patch.dict(os.environ, {'GROK_API_KEY': 'grok-valid_key_123'}):
            key = self.config.get_grok_api_key()
            assert key == 'grok-valid_key_123'
    
    def test_get_grok_api_key_placeholder(self):
        """Test that placeholder Grok key returns None."""
        with patch.dict(os.environ, {'GROK_API_KEY': 'your_key_here'}):
            key = self.config.get_grok_api_key()
            assert key is None
    
    def test_get_grok_api_key_missing(self):
        """Test that missing Grok key returns None."""
        with patch.dict(os.environ, {}, clear=True):
            key = self.config.get_grok_api_key()
            assert key is None
    
    def test_get_log_level_default(self):
        """Test getting default log level."""
        with patch.dict(os.environ, {}, clear=True):
            level = self.config.get_log_level()
            assert level == 'INFO'
    
    def test_get_log_level_custom(self):
        """Test getting custom log level."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'debug'}):
            level = self.config.get_log_level()
            assert level == 'DEBUG'  # Should be uppercase
    
    def test_get_openai_model_default(self):
        """Test getting default OpenAI model."""
        with patch.dict(os.environ, {}, clear=True):
            model = self.config.get_openai_model()
            assert model == 'gpt-4o'
    
    def test_get_openai_model_custom(self):
        """Test getting custom OpenAI model."""
        with patch.dict(os.environ, {'OPENAI_MODEL': 'gpt-3.5-turbo'}):
            model = self.config.get_openai_model()
            assert model == 'gpt-3.5-turbo'
    
    def test_get_openai_temperature_default(self):
        """Test getting default OpenAI temperature."""
        with patch.dict(os.environ, {}, clear=True):
            temp = self.config.get_openai_temperature()
            assert temp == 0.7
    
    def test_get_openai_temperature_custom(self):
        """Test getting custom OpenAI temperature."""
        with patch.dict(os.environ, {'OPENAI_TEMPERATURE': '0.5'}):
            temp = self.config.get_openai_temperature()
            assert temp == 0.5
    
    def test_get_openai_temperature_invalid(self):
        """Test getting invalid OpenAI temperature falls back to default."""
        with patch.dict(os.environ, {'OPENAI_TEMPERATURE': 'invalid'}):
            with patch('src.config.logger') as mock_logger:
                temp = self.config.get_openai_temperature()
                assert temp == 0.7  # Default
                mock_logger.warning.assert_called()
    
    def test_get_openai_max_tokens_default(self):
        """Test getting default OpenAI max tokens."""
        with patch.dict(os.environ, {}, clear=True):
            tokens = self.config.get_openai_max_tokens()
            assert tokens == 4000
    
    def test_get_openai_max_tokens_custom(self):
        """Test getting custom OpenAI max tokens."""
        with patch.dict(os.environ, {'OPENAI_MAX_TOKENS': '2000'}):
            tokens = self.config.get_openai_max_tokens()
            assert tokens == 2000
    
    def test_get_openai_max_tokens_invalid(self):
        """Test getting invalid OpenAI max tokens falls back to default."""
        with patch.dict(os.environ, {'OPENAI_MAX_TOKENS': 'invalid'}):
            with patch('src.config.logger') as mock_logger:
                tokens = self.config.get_openai_max_tokens()
                assert tokens == 4000  # Default
                mock_logger.warning.assert_called()
    
    def test_get_perplexity_model_default(self):
        """Test getting default Perplexity model."""
        with patch.dict(os.environ, {}, clear=True):
            model = self.config.get_perplexity_model()
            assert model == 'llama-3.1-sonar-large-128k-online'
    
    def test_get_perplexity_model_custom(self):
        """Test getting custom Perplexity model."""
        with patch.dict(os.environ, {'PERPLEXITY_MODEL': 'custom-model'}):
            model = self.config.get_perplexity_model()
            assert model == 'custom-model'
    
    def test_get_perplexity_max_tokens_default(self):
        """Test getting default Perplexity max tokens."""
        with patch.dict(os.environ, {}, clear=True):
            tokens = self.config.get_perplexity_max_tokens()
            assert tokens == 4000
    
    def test_get_perplexity_max_tokens_custom(self):
        """Test getting custom Perplexity max tokens."""
        with patch.dict(os.environ, {'PERPLEXITY_MAX_TOKENS': '3000'}):
            tokens = self.config.get_perplexity_max_tokens()
            assert tokens == 3000
    
    def test_get_perplexity_max_tokens_invalid(self):
        """Test getting invalid Perplexity max tokens falls back to default."""
        with patch.dict(os.environ, {'PERPLEXITY_MAX_TOKENS': 'invalid'}):
            with patch('src.config.logger') as mock_logger:
                tokens = self.config.get_perplexity_max_tokens()
                assert tokens == 4000  # Default
                mock_logger.warning.assert_called()
    
    def test_get_env_var_with_default(self):
        """Test getting environment variable with default value."""
        with patch.dict(os.environ, {}, clear=True):
            value = self.config.get_env_var('NONEXISTENT_VAR', 'default_value')
            assert value == 'default_value'
    
    def test_get_env_var_with_cast(self):
        """Test getting environment variable with casting."""
        with patch.dict(os.environ, {'TEST_INT': '42'}):
            value = self.config.get_env_var('TEST_INT', 0, cast=int)
            assert value == 42
            assert isinstance(value, int)
    
    def test_get_env_var_cast_failure(self):
        """Test environment variable casting failure."""
        with patch.dict(os.environ, {'TEST_INT': 'not_a_number'}):
            with patch('src.config.logger') as mock_logger:
                value = self.config.get_env_var('TEST_INT', 0, cast=int)
                assert value == 0  # Default value
                mock_logger.warning.assert_called()
    
    def test_get_env_var_existing_value(self):
        """Test getting existing environment variable."""
        with patch.dict(os.environ, {'EXISTING_VAR': 'existing_value'}):
            value = self.config.get_env_var('EXISTING_VAR', 'default')
            assert value == 'existing_value'
    
    def test_validate_stage1_requirements_true(self):
        """Test stage 1 validation with valid Discord token."""
        with patch.object(self.config, 'get_discord_token', return_value='valid_token'):
            assert self.config.validate_stage1_requirements() is True
    
    def test_validate_stage1_requirements_false(self):
        """Test stage 1 validation with invalid Discord token."""
        with patch.object(self.config, 'get_discord_token', return_value=None):
            assert self.config.validate_stage1_requirements() is False
    
    def test_validate_stage2_requirements_true(self):
        """Test stage 2 validation with valid tokens."""
        with patch.object(self.config, 'validate_stage1_requirements', return_value=True):
            with patch.object(self.config, 'get_openai_api_key', return_value='valid_key'):
                assert self.config.validate_stage2_requirements() is True
    
    def test_validate_stage2_requirements_false_stage1(self):
        """Test stage 2 validation fails when stage 1 fails."""
        with patch.object(self.config, 'validate_stage1_requirements', return_value=False):
            with patch.object(self.config, 'get_openai_api_key', return_value='valid_key'):
                assert self.config.validate_stage2_requirements() is False
    
    def test_validate_stage2_requirements_false_openai(self):
        """Test stage 2 validation fails when OpenAI key is missing."""
        with patch.object(self.config, 'validate_stage1_requirements', return_value=True):
            with patch.object(self.config, 'get_openai_api_key', return_value=None):
                assert self.config.validate_stage2_requirements() is False
    
    def test_validate_stage3_requirements_true(self):
        """Test stage 3 validation with valid tokens."""
        with patch.object(self.config, 'validate_stage2_requirements', return_value=True):
            with patch.object(self.config, 'get_perplexity_api_key', return_value='valid_key'):
                assert self.config.validate_stage3_requirements() is True
    
    def test_validate_stage3_requirements_false_stage2(self):
        """Test stage 3 validation fails when stage 2 fails."""
        with patch.object(self.config, 'validate_stage2_requirements', return_value=False):
            with patch.object(self.config, 'get_perplexity_api_key', return_value='valid_key'):
                assert self.config.validate_stage3_requirements() is False
    
    def test_validate_stage3_requirements_false_perplexity(self):
        """Test stage 3 validation fails when Perplexity key is missing."""
        with patch.object(self.config, 'validate_stage2_requirements', return_value=True):
            with patch.object(self.config, 'get_perplexity_api_key', return_value=None):
                assert self.config.validate_stage3_requirements() is False
    
    def test_get_current_stage_all_valid(self):
        """Test get_current_stage with all requirements met."""
        with patch.object(self.config, 'validate_stage3_requirements', return_value=True):
            assert self.config.get_current_stage() == 4
    
    def test_get_current_stage_partial(self):
        """Test get_current_stage with partial requirements."""
        with patch.object(self.config, 'validate_stage3_requirements', return_value=False):
            with patch.object(self.config, 'validate_stage2_requirements', return_value=True):
                assert self.config.get_current_stage() == 2
    
    def test_get_current_stage_none(self):
        """Test get_current_stage with no requirements met."""
        with patch.object(self.config, 'validate_stage1_requirements', return_value=False):
            assert self.config.get_current_stage() == 0
    
    def test_get_stage_info_all_true(self):
        """Test get_stage_info with all stages valid."""
        with patch.object(self.config, 'validate_stage1_requirements', return_value=True):
            with patch.object(self.config, 'validate_stage2_requirements', return_value=True):
                with patch.object(self.config, 'validate_stage3_requirements', return_value=True):
                    info = self.config.get_stage_info()
                    
                    assert info['stage1_discord'] is True
                    assert info['stage2_openai'] is True
                    assert info['stage3_perplexity'] is True
    
    def test_get_stage_info_mixed(self):
        """Test get_stage_info with mixed stage validity."""
        with patch.object(self.config, 'validate_stage1_requirements', return_value=True):
            with patch.object(self.config, 'validate_stage2_requirements', return_value=False):
                with patch.object(self.config, 'validate_stage3_requirements', return_value=False):
                    info = self.config.get_stage_info()
                    
                    assert info['stage1_discord'] is True
                    assert info['stage2_openai'] is False
                    assert info['stage3_perplexity'] is False


class TestConfigIntegration:
    """Integration tests for Config class."""
    
    def test_config_singleton_behavior(self):
        """Test that config behaves as expected in real usage."""
        config1 = Config()
        config2 = Config()
        
        # Both instances should work independently (not singleton pattern)
        assert config1 is not config2
    

    def test_config_error_recovery(self):
        """Test that config handles various error conditions gracefully."""
        # Test with problematic environment variables
        problematic_vars = {
            'OPENAI_TEMPERATURE': 'not_a_float',
            'OPENAI_MAX_TOKENS': 'not_an_int',
            'PERPLEXITY_MAX_TOKENS': '-999',  # Invalid negative value
            'LOG_LEVEL': '',  # Empty value
        }
        
        with patch.dict(os.environ, problematic_vars):
            config = Config()
            
            # Should fall back to defaults gracefully
            assert config.get_openai_temperature() == 0.7  # Default
            assert config.get_openai_max_tokens() == 4000  # Default
            assert config.get_perplexity_max_tokens() == 4000  # Default
            assert config.get_log_level() == ''  # Returns the actual value, even if empty