import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger

# --- 모듈 임포트 (수정된 부분) ---
from src.config import config
from src.models import KeywordResult, SearchResult

# 각 기능의 핵심 함수만 import 하도록 수정
try:
    from src.keyword_generator import generate_keywords_for_topic

    KEYWORD_GENERATION_AVAILABLE = True
    logger.info("✅ 키워드 생성 모듈 로드 완료")
except ImportError as e:
    KEYWORD_GENERATION_AVAILABLE = False
    logger.warning(f"⚠️ 키워드 생성 모듈 로드 실패: {e}")

try:
    # 'issue_searcher'에서는 검색 실행 함수만 가져옵니다.
    from src.issue_searcher import search_issues_for_keywords
    # 'reporting'에서는 모든 보고서 관련 함수를 가져옵니다.
    from src.reporting import (
        format_search_summary,
        create_detailed_report_from_search_result,
        save_report_to_file
    )

    ISSUE_SEARCH_AVAILABLE = True
    logger.info("✅ 이슈 검색 및 보고서 모듈 로드 완료")
except ImportError as e:
    ISSUE_SEARCH_AVAILABLE = False
    logger.warning(f"⚠️ 이슈 검색 및 보고서 모듈 로드 실패: {e}")

# --- 로깅 설정 ---
# (기존과 동일)
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
           level="DEBUG", colorize=True)
log_file = "logs/bot.log"
if os.path.exists(log_file):
    try:
        os.remove(log_file)
    except OSError as e:
        logger.error(f"로그 파일 삭제 실패: {e}")
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
           level="INFO", encoding="utf-8")
error_log_file = "logs/error.log"
if os.path.exists(error_log_file):
    try:
        os.remove(error_log_file)
    except OSError as e:
        logger.error(f"에러 로그 파일 삭제 실패: {e}")
logger.add(error_log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
           level="ERROR", encoding="utf-8")
logger.info("🚀 봇 시작 중...")
current_stage = config.get_current_stage()
logger.info(f"⚙️ 현재 실행 가능 단계: {current_stage}단계")
if config.is_development_mode(): logger.info("🔧 개발 모드로 실행 중")

# --- 봇 클래스 및 이벤트 핸들러 ---
# (기존과 동일)
intents = discord.Intents.default()
intents.message_content = True


class IssueMonitorBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        logger.info("🤖 IssueMonitorBot 인스턴스 생성됨")

    async def setup_hook(self):
        logger.info("⚙️ 봇 셋업 시작")
        try:
            synced = await self.tree.sync()
            logger.success(f"✅ 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")
        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")

    async def on_ready(self):
        logger.success(f"🎉 {self.user}가 Discord에 연결되었습니다!")
        logger.info(f"📊 봇이 {len(self.guilds)}개 서버에 참여 중")
        status_message = f"/monitor commands (Stage {current_stage})"
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=status_message))
        logger.info(f"👀 봇 상태 설정: {status_message}")

    async def on_error(self, event, *args, **kwargs):
        logger.error(f"❌ 이벤트 오류 ({event}): {args}")

    async def close(self):
        logger.info("🛑 봇 종료 중...")
        await super().close()


bot = IssueMonitorBot()


# --- 헬퍼 함수 ---
# (기존과 동일)
def parse_time_period(period_str: str) -> tuple[datetime, str]:
    period_str = period_str.strip().lower()
    now = datetime.now()
    match = re.match(r'(\d+)\s*(일|주일|개월|달|시간)', period_str)
    if not match: return now - timedelta(weeks=1), "최근 1주일"
    number = int(match.group(1))
    unit = match.group(2)
    if unit in ['일']: return now - timedelta(days=number), f"최근 {number}일"
    if unit in ['주일']: return now - timedelta(weeks=number), f"최근 {number}주일"
    if unit in ['개월', '달']: return now - timedelta(days=number * 30), f"최근 {number}개월"
    if unit in ['시간']: return now - timedelta(hours=number), f"최근 {number}시간"
    return now - timedelta(weeks=1), "최근 1주일"


def validate_topic(topic: str) -> bool:
    return topic is not None and len(topic.strip()) >= 2


# --- 슬래시 명령어 ---
@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링합니다")
async def monitor_command(interaction: discord.Interaction, 주제: str, 기간: str = "1주일", 세부분석: bool = True):
    user = interaction.user
    guild = interaction.guild
    logger.info(f"📝 /monitor: user={user.name}, guild={guild.name}, topic='{주제}', period='{기간}', details={세부분석}")
    await interaction.response.defer(thinking=True)
    try:
        if not validate_topic(주제):
            await interaction.followup.send("❌ 주제를 2글자 이상 입력해주세요.", ephemeral=True)
            return

        _, period_description = parse_time_period(기간)

        embed = discord.Embed(title="🔍 이슈 모니터링 시작", description=f"**주제**: {주제}\n**기간**: {period_description}",
                              color=0x00aaff, timestamp=datetime.now())
        await interaction.followup.send(embed=embed)

        if not KEYWORD_GENERATION_AVAILABLE or not ISSUE_SEARCH_AVAILABLE:
            await interaction.followup.send("⚠️ 봇의 일부 기능이 로드되지 않았습니다. 설정을 확인해주세요.", ephemeral=True)
            return

        keyword_result = await generate_keywords_for_topic(주제)
        search_result = await search_issues_for_keywords(keyword_result, period_description, collect_details=세부분석)

        success_embed = discord.Embed(title=f"✅ 이슈 모니터링 완료: {주제}", color=0x00ff00)
        search_summary = format_search_summary(search_result)
        success_embed.add_field(name="📈 분석 결과 요약", value=search_summary, inline=False)

        if search_result.detailed_issues_count > 0:
            report_content = create_detailed_report_from_search_result(search_result)
            file_path = save_report_to_file(report_content, 주제)
            with open(file_path, 'rb') as f:
                await interaction.followup.send(embed=success_embed,
                                                file=discord.File(f, filename=os.path.basename(file_path)))
        else:
            await interaction.followup.send(embed=success_embed)

    except Exception as e:
        logger.error(f"💥 /monitor 오류: {e}", exc_info=True)
        error_embed = discord.Embed(title="❌ 시스템 오류 발생", description=f"요청 처리 중 문제가 발생했습니다.\n`오류: {e}`", color=0xff0000)
        if interaction.is_deferred():
            await interaction.followup.send(embed=error_embed, ephemeral=True)
        else:
            await interaction.response.send_message(embed=error_embed, ephemeral=True)


# (help, status 명령어 및 run_bot 함수는 기존과 동일)
@bot.tree.command(name="help", description="봇 사용법을 안내합니다")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="🤖 이슈 모니터링 봇 사용법", color=0x0099ff)
    embed.add_field(name="`/monitor`", value="`주제`, `기간`, `세부분석` 옵션을 사용하여 특정 주제의 이슈를 모니터링합니다.", inline=False)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="status", description="봇 시스템 상태를 확인합니다")
async def status_command(interaction: discord.Interaction):
    stage = config.get_current_stage()
    embed = discord.Embed(title="📊 시스템 상태", description=f"현재 실행 가능한 최고 단계: **{stage}단계**", color=0x00ff00)
    await interaction.response.send_message(embed=embed)


def run_bot():
    discord_token = config.get_discord_token()
    if not discord_token:
        logger.critical("❌ Discord 봇 토큰이 없습니다. .env 파일을 확인해주세요!")
        return
    try:
        logger.info("🚀 Discord 봇을 시작합니다...")
        bot.run(discord_token, log_handler=None)
    except Exception as e:
        logger.critical(f"💥 봇 실행 실패: {e}", exc_info=True)


if __name__ == "__main__":
    run_bot()