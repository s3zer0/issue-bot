import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
import tempfile
from loguru import logger

# 💡 [수정] 모든 import 구문에 'src.'를 추가하여 경로를 명확히 함
from src.config import config

# 키워드 생성기 import
try:
    from src.keyword_generator import create_keyword_generator, generate_keywords_for_topic

    KEYWORD_GENERATION_AVAILABLE = True
    logger.info("✅ 키워드 생성 모듈 로드 완료")
except ImportError as e:
    KEYWORD_GENERATION_AVAILABLE = False
    logger.warning(f"⚠️ 키워드 생성 모듈 로드 실패: {e}")

# 이슈 검색기 import
try:
    from src.issue_searcher import (
        create_issue_searcher,
        search_issues_for_keywords,
        create_detailed_report_from_search_result
    )

    ISSUE_SEARCH_AVAILABLE = True
    logger.info("✅ 이슈 검색 모듈 로드 완료")
except ImportError as e:
    ISSUE_SEARCH_AVAILABLE = False
    logger.warning(f"⚠️ 이슈 검색 모듈 로드 실패: {e}")

# 로그 설정
os.makedirs("logs", exist_ok=True)
logger.remove()

# 콘솔 로그
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

# 파일 로그
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

# 에러 로그
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

# 설정 상태 확인
current_stage = config.get_current_stage()
logger.info(f"⚙️ 현재 실행 가능 단계: {current_stage}단계")

if config.is_development_mode():
    logger.info("🔧 개발 모드로 실행 중")

# 인텐트 설정
intents = discord.Intents.default()
intents.message_content = True


class IssueMonitorBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        logger.info("🤖 IssueMonitorBot 인스턴스 생성됨")

    async def setup_hook(self):
        """봇 시작 시 초기화 작업"""
        logger.info("⚙️ 봇 셋업 시작")
        try:
            synced = await self.tree.sync()
            logger.success(f"✅ 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")
        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")

    async def on_ready(self):
        """봇이 준비되면 실행"""
        logger.success(f"🎉 {self.user}가 Discord에 연결되었습니다!")
        logger.info(f"📊 봇이 {len(self.guilds)}개 서버에 참여 중")

        for guild in self.guilds:
            logger.info(f"  📋 서버: {guild.name} (ID: {guild.id}, 멤버: {guild.member_count}명)")

        # 봇 상태 설정
        status_message = f"/monitor commands (Stage {current_stage})"
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=status_message
            )
        )
        logger.info(f"👀 봇 상태 설정: {status_message}")

    async def on_guild_join(self, guild):
        logger.info(f"🆕 새 서버 참여: {guild.name} (ID: {guild.id})")

    async def on_guild_remove(self, guild):
        logger.info(f"👋 서버 퇴장: {guild.name} (ID: {guild.id})")

    async def on_command_error(self, ctx, error):
        logger.error(f"❌ 명령어 오류: {error}")

    async def on_error(self, event, *args, **kwargs):
        logger.error(f"❌ 이벤트 오류 ({event}): {args}")

    async def close(self):
        logger.info("🛑 봇 종료 중...")
        await super().close()


bot = IssueMonitorBot()


def parse_time_period(period_str: str) -> tuple[datetime, str]:
    """시간 기간 문자열을 파싱하여 시작 날짜와 설명을 반환"""
    period_str = period_str.strip().lower()
    now = datetime.now()

    match = re.match(r'(\d+)\s*(일|주일|개월|달|시간)', period_str)

    if not match:
        return now - timedelta(weeks=1), "최근 1주일"

    number = int(match.group(1))
    unit = match.group(2)

    if unit in ['일']:
        start_date = now - timedelta(days=number)
        description = f"최근 {number}일"
    elif unit in ['주일']:
        start_date = now - timedelta(weeks=number)
        description = f"최근 {number}주일"
    elif unit in ['개월', '달']:
        start_date = now - timedelta(days=number * 30)
        description = f"최근 {number}개월"
    elif unit in ['시간']:
        start_date = now - timedelta(hours=number)
        description = f"최근 {number}시간"
    else:
        start_date = now - timedelta(weeks=1)
        description = "최근 1주일"

    return start_date, description


def validate_topic(topic: str) -> bool:
    """주제 입력값 검증"""
    return topic is not None and len(topic.strip()) >= 2


def validate_period(period: str) -> bool:
    """기간 입력값 검증"""
    if not period:
        return True

    import re
    pattern = r'(\d+)\s*(일|주일|개월|달|시간)'
    return bool(re.match(pattern, period.strip().lower()))


@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링합니다")
async def monitor_command(
        interaction: discord.Interaction,
        주제: str,
        기간: str = "1주일",
        세부분석: bool = True
):
    """이슈 모니터링 메인 명령어"""
    user = interaction.user
    guild = interaction.guild
    logger.info(
        f"📝 /monitor 명령어 실행: 사용자={user.name}, 서버={guild.name}, 주제='{주제}', 기간='{기간}', 세부분석={세부분석}")

    await interaction.response.defer(thinking=True)

    try:
        if not validate_topic(주제):
            logger.warning(f"❌ 잘못된 주제 입력: '{주제}' (사용자: {user.name})")
            await interaction.followup.send("❌ 주제를 2글자 이상 입력해주세요.", ephemeral=True)
            return

        start_date, period_description = parse_time_period(기간)
        available_stage = config.get_current_stage()

        embed = discord.Embed(
            title="🔍 이슈 모니터링 시작",
            description=f"**주제**: {주제}\n**기간**: {period_description}\n**세부분석**: {'활성화' if 세부분석 else '비활성화'}",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        await interaction.followup.send(embed=embed)

        if available_stage >= 2 and KEYWORD_GENERATION_AVAILABLE:
            keyword_result = await generate_keywords_for_topic(주제)

            if available_stage >= 3 and ISSUE_SEARCH_AVAILABLE:
                search_result = await search_issues_for_keywords(keyword_result, period_description,
                                                                 collect_details=세부분석)

                success_embed = discord.Embed(title=f"✅ {available_stage}단계 완료: 이슈 모니터링",
                                              description=f"주제 '{주제}'에 대한 모니터링 완료.", color=0x00ff00)

                # 결과 요약 추가
                search_summary = create_issue_searcher().format_search_summary(search_result)
                success_embed.add_field(name="📈 분석 결과", value=search_summary, inline=False)

                # 상세 보고서 파일 첨부
                if search_result.detailed_issues_count > 0:
                    detailed_report = create_detailed_report_from_search_result(search_result)
                    reports_dir = "reports"
                    os.makedirs(reports_dir, exist_ok=True)

                    filename = f"issue_report_{주제.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    file_path = os.path.join(reports_dir, filename)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(detailed_report)

                    # Discord에 파일 전송
                    with open(file_path, 'rb') as f:
                        await interaction.followup.send(embed=success_embed, file=discord.File(f, filename=filename))
                else:
                    await interaction.followup.send(embed=success_embed)

            else:  # 2단계만 가능
                limitation_embed = discord.Embed(title="⚠️ 기능 제한 (2단계 완료)",
                                                 description="키워드 생성은 완료되었으나, 이슈 검색을 위해 추가 설정이 필요합니다.", color=0xffaa00)
                await interaction.followup.send(embed=limitation_embed)
        else:  # 1단계 또는 0단계
            limitation_embed = discord.Embed(title="⚠️ 기능 제한",
                                             description=f"현재 {available_stage}단계까지만 설정되어 기능 실행이 어렵습니다.",
                                             color=0xffaa00)
            await interaction.followup.send(embed=limitation_embed)

    except Exception as e:
        logger.error(f"💥 monitor 명령어 실행 중 오류: {e}", exc_info=True)
        await interaction.followup.send(f"❌ 시스템 오류 발생: {e}", ephemeral=True)


@bot.tree.command(name="help", description="봇 사용법을 안내합니다")
async def help_command(interaction: discord.Interaction):
    """도움말 명령어"""
    embed = discord.Embed(title="🤖 이슈 모니터링 봇 사용법", color=0x0099ff)
    embed.add_field(name="`/monitor`", value="`주제`, `기간`, `세부분석` 옵션을 사용하여 특정 주제의 이슈를 모니터링합니다.", inline=False)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="status", description="봇 시스템 상태를 확인합니다")
async def status_command(interaction: discord.Interaction):
    """시스템 상태 확인 명령어"""
    current_stage = config.get_current_stage()
    embed = discord.Embed(title="📊 시스템 상태", description=f"현재 실행 가능한 최고 단계: **{current_stage}단계**", color=0x00ff00)
    await interaction.response.send_message(embed=embed)


def run_bot():
    """봇 실행 함수"""
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