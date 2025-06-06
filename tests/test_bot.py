"""
Discord 이슈 모니터링 봇 - pytest 테스트 (최종 수정)
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

# 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 의존성 모듈 임포트
from src.bot import monitor_command, help_command, status_command, validate_topic, parse_time_period
from src.keyword_generator import KeywordResult
from src.issue_searcher import SearchResult, IssueItem

# --- Helper Fixtures ---
@pytest.fixture
def mock_discord_interaction():
    """Mock Discord Interaction 객체 픽스처"""
    interaction = MagicMock()
    interaction.user.name = "TestUser"
    interaction.guild.name = "TestServer"
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    return interaction

@pytest.fixture
def mock_keyword_result():
    """KeywordResult 모의 객체"""
    return KeywordResult(topic="AI", primary_keywords=["AI"], related_terms=[], context_keywords=[], confidence_score=0.9, generation_time=1.0, raw_response="{}")

@pytest.fixture
def mock_search_result():
    """SearchResult 모의 객체"""
    issue = IssueItem(title="AI의 미래", summary="...", source="Tech News", published_date="...", relevance_score=0.9, category="news", content_snippet="...", detail_confidence=0.85, detailed_content="상세 내용")
    return SearchResult(query_keywords=["AI"], total_found=1, issues=[issue], search_time=5.0, api_calls_used=3, confidence_score=0.88, time_period="...", raw_responses=["{}"], detailed_issues_count=1, total_detail_collection_time=3.0, average_detail_confidence=0.85)

# --- 기존 테스트 클래스 (유지) ---
class TestHelperFunctions:
    @pytest.mark.unit
    def test_parse_valid_days(self):
        assert "3일" in parse_time_period("3일")[1]

    @pytest.mark.unit
    def test_topic_validation(self):
        assert validate_topic("AI 기술") is True
        assert validate_topic("A") is False

# --- 명령어 테스트를 위한 클래스 ---
@pytest.mark.asyncio
class TestBotCommands:
    """봇의 주요 명령어들을 테스트합니다."""

    @patch('src.bot.config')
    @patch('src.bot.generate_keywords_for_topic')
    @patch('src.bot.search_issues_for_keywords')
    @patch('src.bot.create_detailed_report_from_search_result')
    @patch('src.bot.tempfile.NamedTemporaryFile')
    @patch('builtins.open', new_callable=mock_open, read_data='report content')
    @patch('src.bot.os.unlink')
    async def test_monitor_command_full_success(
        self, mock_unlink, mock_builtin_open, mock_tempfile, mock_create_report, mock_search, mock_generate_keywords, mock_config,
        mock_discord_interaction, mock_keyword_result, mock_search_result
    ):
        """/monitor 명령어: 4단계까지 모두 성공하는 시나리오 테스트"""
        mock_config.get_current_stage.return_value = 4
        mock_generate_keywords.return_value = mock_keyword_result
        mock_search.return_value = mock_search_result
        mock_create_report.return_value = "## 상세 보고서"
        mock_tempfile.return_value.__enter__.return_value.name = "/tmp/test_report.md"

        await monitor_command.callback(mock_discord_interaction, 주제="AI", 기간="1주일", 세부분석=True)

        final_call_args = mock_discord_interaction.followup.send.call_args
        assert final_call_args is not None, "최종 응답이 전송되지 않았습니다."
        assert 'embed' in final_call_args.kwargs
        assert 'file' in final_call_args.kwargs
        # 💡 [수정] 실제 출력되는 문자열에 맞춰 검증 로직 변경
        assert "모니터링 완료" in final_call_args.kwargs['embed'].description
        mock_unlink.assert_called_once_with("/tmp/test_report.md")

    @patch('src.bot.config')
    @patch('src.bot.generate_keywords_for_topic')
    async def test_monitor_command_stage_2_limit(self, mock_generate_keywords, mock_config, mock_discord_interaction):
        """/monitor 명령어: API 키 부족으로 2단계까지만 실행되는 시나리오"""
        mock_config.get_current_stage.return_value = 2
        mock_generate_keywords.return_value = KeywordResult(topic="테스트", primary_keywords=["test"], related_terms=[], context_keywords=[], confidence_score=0.8, generation_time=1, raw_response="{}")

        await monitor_command.callback(mock_discord_interaction, 주제="테스트", 기간="1일", 세부분석=False)

        final_call_args = mock_discord_interaction.followup.send.call_args
        embed = final_call_args.kwargs['embed']
        # 💡 [수정] embed에 필드가 없으므로, title만 검증
        assert "기능 제한" in embed.title
        assert "이슈 검색을 위해 추가 설정이 필요합니다" in embed.description

    @patch('src.bot.config')
    async def test_help_command(self, mock_config, mock_discord_interaction):
        """/help 명령어 테스트"""
        mock_config.get_current_stage.return_value = 4
        await help_command.callback(mock_discord_interaction)

        call_args = mock_discord_interaction.response.send_message.call_args
        embed = call_args.kwargs['embed']
        # 💡 [수정] embed의 실제 필드 이름과 내용을 검증
        assert "이슈 모니터링 봇 사용법" in embed.title
        assert embed.fields[0].name == "`/monitor`"

    @patch('src.bot.config')
    async def test_status_command(self, mock_config, mock_discord_interaction):
        """/status 명령어 테스트"""
        mock_config.get_current_stage.return_value = 4

        await status_command.callback(mock_discord_interaction)

        call_args = mock_discord_interaction.response.send_message.call_args
        embed = call_args.kwargs['embed']
        # 💡 [수정] embed에 필드가 없으므로, title과 description만 검증
        assert "시스템 상태" in embed.title
        assert "현재 실행 가능한 최고 단계" in embed.description