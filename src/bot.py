import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger

# --- 모듈 임포트 ---
from src.models import KeywordResult, SearchResult
from src.config import config

try:
    from src.keyword_generator import generate_keywords_for_topic

    KEYWORD_GENERATION_AVAILABLE = True
    logger.info("✅ 키워드 생성 모듈 로드 완료")
except ImportError as e:
    KEYWORD_GENERATION_AVAILABLE = False
    logger.warning(f"⚠️ 키워드 생성 모듈 로드 실패: {e}")

try:
    # 환각 탐지 기능이 포함된 검색기 및 보고서 모듈 import
    from src.hallucination_detector import RePPLEnhancedIssueSearcher
    from src.reporting import (
        format_search_summary,
        create_detailed_report_from_search_result,
        save_report_to_file
    )

    ISSUE_SEARCH_AVAILABLE = True
    logger.info("✅ 이슈 검색, 환각 탐지 및 보고서 모듈 로드 완료")
except ImportError as e:
    ISSUE_SEARCH_AVAILABLE = False
    logger.warning(f"⚠️ 이슈 검색 관련 모듈 로드 실패: {e}")

# --- 로깅 설정 ---
os.makedirs("logs", exist_ok=True)
logger.remove()
# 콘솔 로그 설정
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True
)
# 파일 로그 설정
log_file = "logs/bot.log"
if os.path.exists(log_file):
    try:
        os.remove(log_file)
    except OSError as e:
        logger.error(f"로그 파일 삭제 실패: {e}")
logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    encoding="utf-8"
)
# 에러 로그 설정
error_log_file = "logs/error.log"
if os.path.exists(error_log_file):
    try:
        os.remove(error_log_file)
    except OSError as e:
        logger.error(f"에러 로그 파일 삭제 실패: {e}")
logger.add(
    error_log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    encoding="utf-8"
)

logger.info("🚀 봇 시작 중...")
current_stage = config.get_current_stage()
logger.info(f"⚙️ 현재 실행 가능 단계: {current_stage}단계")
if config.is_development_mode():
    logger.info("🔧 개발 모드로 실행 중")

# --- 봇 클래스 및 이벤트 핸들러 ---
intents = discord.Intents.default()
intents.message_content = True


class IssueMonitorBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        logger.info("🤖 IssueMonitorBot 인스턴스 생성됨")

    async def setup_hook(self):
        """봇 시작 시 슬래시 명령어 동기화"""
        logger.info("⚙️ 봇 셋업 시작: 슬래시 명령어 동기화 시도...")
        try:
            synced = await self.tree.sync()
            logger.success(f"✅ 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")
        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")

    async def on_ready(self):
        """봇이 준비되면 실행되는 이벤트"""
        logger.success(f"🎉 {self.user}가 Discord에 성공적으로 연결되었습니다!")
        logger.info(f"📊 봇이 {len(self.guilds)}개 서버에 참여 중입니다.")

        status_message = f"/monitor (Stage {current_stage} 활성화)"
        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name=status_message)
        )
        logger.info(f"👀 봇 상태 설정: '{status_message}'")

    async def on_error(self, event, *args, **kwargs):
        """예상치 못한 이벤트 오류 발생 시 로깅"""
        logger.error(f"❌ 처리되지 않은 이벤트 오류 발생 ({event}): {args} {kwargs}")


bot = IssueMonitorBot()


# --- 헬퍼 함수 ---
def parse_time_period(period_str: str) -> tuple[datetime, str]:
    """'1주일', '3일' 등 시간 문자열을 파싱합니다."""
    period_str = period_str.strip().lower()
    now = datetime.now()
    match = re.match(r'(\d+)\s*(일|주일|개월|달|시간)', period_str)

    if not match: return now - timedelta(weeks=1), "최근 1주일"

    number = int(match.group(1))
    unit = match.group(2)

    if unit == '일': return now - timedelta(days=number), f"최근 {number}일"
    if unit == '주일': return now - timedelta(weeks=number), f"최근 {number}주일"
    if unit in ['개월', '달']: return now - timedelta(days=number * 30), f"최근 {number}개월"
    if unit == '시간': return now - timedelta(hours=number), f"최근 {number}시간"

    return now - timedelta(weeks=1), "최근 1주일"  # 기본값


def validate_topic(topic: str) -> bool:
    """주제 입력값이 유효한지 검사합니다."""
    return topic is not None and len(topic.strip()) >= 2


@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링하고 환각 현상을 검증합니다.")
async def monitor_command(interaction: discord.Interaction, 주제: str, 기간: str = "1주일"):
    user = interaction.user
    logger.info(f"📝 /monitor 명령어 수신: 사용자='{user.name}', 주제='{주제}', 기간='{기간}'")
    await interaction.response.defer(thinking=True)

    try:
        # ... (입력값 검증 및 초기 embed 전송 로직은 기존과 동일) ...
        _, period_description = parse_time_period(기간)
        embed = discord.Embed(
            title="🔍 이슈 모니터링 시작 (환각 탐지 활성화)",
            description=f"**주제**: {주제}\n**기간**: {period_description}",
            color=0x00aaff,
            timestamp=datetime.now()
        )
        await interaction.followup.send(embed=embed)

        # 1. 키워드 생성
        keyword_result = await generate_keywords_for_topic(주제)

        # 2. RePPL 강화 검색기 실행
        enhanced_searcher = RePPLEnhancedIssueSearcher()
        search_result = await enhanced_searcher.search_with_validation(keyword_result, period_description)

        # 3. 결과 보고
        success_embed = discord.Embed(title=f"✅ 이슈 모니터링 완료: {주제}", color=0x00ff00)
        search_summary = format_search_summary(search_result)
        success_embed.add_field(name="📈 분석 결과 요약 (환각 탐지 완료)", value=search_summary, inline=False)

        # 💡 [수정] 상세 이슈 개수(detailed_issues_count)와 관계없이 항상 보고서를 생성하고 파일로 전송합니다.
        report_content = create_detailed_report_from_search_result(search_result)
        file_path = save_report_to_file(report_content, 주제)

        with open(file_path, 'rb') as f:
            discord_file = discord.File(f, filename=os.path.basename(file_path))
            await interaction.followup.send(embed=success_embed, file=discord_file)

    except Exception as e:
        logger.error(f"💥 /monitor 명령어 처리 중 심각한 오류 발생: {e}", exc_info=True)
        error_embed = discord.Embed(
            title="❌ 시스템 오류 발생",
            description=f"요청 처리 중 문제가 발생했습니다. 관리자에게 문의해주세요.\n`오류: {e}`",
            color=0xff0000
        )
        if interaction.is_deferred():
            await interaction.followup.send(embed=error_embed, ephemeral=True)
        else:
            await interaction.response.send_message(embed=error_embed, ephemeral=True)


@bot.tree.command(name="help", description="봇 사용법을 안내합니다.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="🤖 이슈 모니터링 봇 사용법", color=0x0099ff,
                          description="이 봇은 최신 기술 이슈를 모니터링하고 LLM의 환각 현상을 최소화하여 신뢰도 높은 정보를 제공합니다.")
    embed.add_field(name="`/monitor`",
                    value="`주제`와 `기간`을 입력하여 이슈를 검색하고 분석합니다.\n- `주제`: '양자 컴퓨팅', 'AI 반도체' 등\n- `기간`: '3일', '2주일', '1개월' 등",
                    inline=False)
    embed.add_field(name="`/status`", value="봇의 현재 설정 상태와 실행 가능한 단계를 확인합니다.", inline=False)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="status", description="봇 시스템의 현재 설정 상태를 확인합니다.")
async def status_command(interaction: discord.Interaction):
    stage = config.get_current_stage()
    embed = discord.Embed(title="📊 시스템 상태", description=f"현재 실행 가능한 최고 단계는 **{stage}단계**입니다.", color=0x00ff00)
    stage_info = config.get_stage_info()

    embed.add_field(name="1단계: Discord Bot", value="✅" if stage_info['stage1_discord'] else "❌", inline=True)
    embed.add_field(name="2단계: 키워드 생성 (OpenAI)", value="✅" if stage_info['stage2_openai'] else "❌", inline=True)
    embed.add_field(name="3단계/4단계: 이슈 검색 (Perplexity)", value="✅" if stage_info['stage3_perplexity'] else "❌",
                    inline=True)

    await interaction.response.send_message(embed=embed)


# --- 봇 실행 ---
def run_bot():
    """봇을 실행하는 메인 함수"""
    discord_token = config.get_discord_token()
    if not discord_token:
        logger.critical("❌ Discord 봇 토큰이 없습니다. .env 파일을 확인해주세요!")
        return

    try:
        logger.info("🚀 Discord 봇을 시작합니다...")
        bot.run(discord_token, log_handler=None)
    except Exception as e:
        logger.critical(f"💥 봇 실행에 실패했습니다: {e}", exc_info=True)


if __name__ == "__main__":
    run_bot()