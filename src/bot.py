import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger
from config import Config

# logs 디렉토리 생성
os.makedirs("logs", exist_ok=True)

# 로그 설정 - 실행할 때마다 초기화
logger.remove()  # 기본 핸들러 제거

# 콘솔 로그 (컬러풀)
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

# 파일 로그 (실행할 때마다 새로 시작)
log_file = "logs/bot.log"
if os.path.exists(log_file):
    os.remove(log_file)  # 기존 로그 파일 삭제

logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    encoding="utf-8"
)

# 에러만 별도 로그 (실행할 때마다 새로 시작)
error_log_file = "logs/error.log"
if os.path.exists(error_log_file):
    os.remove(error_log_file)  # 기존 에러 로그 파일 삭제

logger.add(
    error_log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    encoding="utf-8"
)

logger.info("🚀 봇 시작 중...")
logger.info(f"📄 로그 파일: {log_file}")
logger.info(f"📄 에러 로그: {error_log_file}")

# 인텐트 설정
intents = discord.Intents.default()
intents.message_content = True


class IssueMonitorBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',  # 슬래시 명령어 주로 사용하지만 백업용
            intents=intents,
            help_command=None
        )
        logger.info("🤖 IssueMonitorBot 인스턴스 생성됨")

    async def setup_hook(self):
        """봇 시작 시 초기화 작업"""
        logger.info("⚙️ 봇 셋업 시작")
        try:
            # 슬래시 명령어 동기화
            synced = await self.tree.sync()
            logger.success(f"✅ 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")
        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")

    async def on_ready(self):
        """봇이 준비되면 실행"""
        logger.success(f"🎉 {self.user}가 Discord에 연결되었습니다!")
        logger.info(f"📊 봇이 {len(self.guilds)}개 서버에 참여 중")

        # 서버 목록 출력
        for guild in self.guilds:
            logger.info(f"  📋 서버: {guild.name} (ID: {guild.id}, 멤버: {guild.member_count}명)")

        # 봇 상태 설정
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for /monitor commands"
            )
        )
        logger.info("👀 봇 상태 설정 완료")

    async def on_guild_join(self, guild):
        """새 서버에 참여했을 때"""
        logger.info(f"🆕 새 서버 참여: {guild.name} (ID: {guild.id})")

    async def on_guild_remove(self, guild):
        """서버에서 나갔을 때"""
        logger.info(f"👋 서버 퇴장: {guild.name} (ID: {guild.id})")

    async def on_command_error(self, ctx, error):
        """명령어 오류 처리"""
        logger.error(f"❌ 명령어 오류: {error}")

    async def on_error(self, event, *args, **kwargs):
        """일반 오류 처리"""
        logger.error(f"❌ 이벤트 오류 ({event}): {args}")

    async def close(self):
        """봇 종료"""
        logger.info("🛑 봇 종료 중...")
        await super().close()


# 봇 인스턴스 생성
bot = IssueMonitorBot()


def parse_time_period(period_str: str) -> tuple[datetime, str]:
    """
    시간 기간 문자열을 파싱하여 시작 날짜와 설명을 반환
    예: "1주일", "3일", "1개월" 등
    """
    period_str = period_str.strip().lower()
    now = datetime.now()

    # 정규식으로 숫자와 단위 추출
    match = re.match(r'(\d+)\s*(일|주일|개월|달|시간)', period_str)

    if not match:
        # 기본값: 1주일
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
        start_date = now - timedelta(days=number * 30)  # 대략적인 계산
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
        return True  # 기본값 사용

    import re
    pattern = r'(\d+)\s*(일|주일|개월|달|시간)'
    return bool(re.match(pattern, period.strip().lower()))


@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링합니다")
async def monitor_command(
        interaction: discord.Interaction,
        주제: str,
        기간: str = "1주일"
):
    """
    이슈 모니터링 메인 명령어
    /monitor 주제:<주제> 기간:<기간>
    """
    user = interaction.user
    guild = interaction.guild
    logger.info(f"📝 /monitor 명령어 실행: 사용자={user.name}#{user.discriminator}, 서버={guild.name}, 주제='{주제}', 기간='{기간}'")

    await interaction.response.defer(thinking=True)  # 처리 시간이 길 수 있으므로

    try:
        # 입력값 검증 및 파싱
        if not validate_topic(주제):
            logger.warning(f"❌ 잘못된 주제 입력: '{주제}' (사용자: {user.name})")
            await interaction.followup.send(
                "❌ 주제를 2글자 이상 입력해주세요.",
                ephemeral=True
            )
            return

        start_date, period_description = parse_time_period(기간)
        logger.info(f"✅ 입력값 검증 완료: 주제='{주제}', 기간='{period_description}'")

        # 초기 응답 (임베드로 정보 정리)
        embed = discord.Embed(
            title="🔍 이슈 모니터링 시작",
            description=f"**주제**: {주제}\n**기간**: {period_description}",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        embed.add_field(
            name="📊 진행 상황",
            value="```\n⏳ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 상세 정보 수집 대기\n⬜ 보고서 생성 대기\n```",
            inline=False
        )
        embed.set_footer(text="예상 소요 시간: 1-3분")

        await interaction.followup.send(embed=embed)
        logger.info(f"📤 초기 응답 전송 완료 (사용자: {user.name})")

        # TODO: 여기서 실제 모니터링 로직 호출
        # 1. LLM 키워드 생성
        # 2. Perplexity API 검색
        # 3. 세부 정보 수집
        # 4. 환각 탐지
        # 5. 보고서 생성

        # 임시 완료 메시지 (실제 구현 전까지)
        logger.info("⏳ 모니터링 시뮬레이션 시작 (2초 대기)")
        await asyncio.sleep(2)  # 시뮬레이션

        success_embed = discord.Embed(
            title="✅ 모니터링 완료 (개발 중)",
            description=f"주제 '{주제}'에 대한 {period_description} 모니터링이 완료되었습니다.",
            color=0x00ff00
        )
        success_embed.add_field(
            name="🚧 개발 상태",
            value="현재 기본 구조만 구현되었습니다.\n다음 단계에서 실제 모니터링 기능을 구현할 예정입니다.",
            inline=False
        )

        await interaction.followup.send(embed=success_embed)
        logger.success(f"✅ 모니터링 완료 응답 전송 (사용자: {user.name})")

        # 로깅
        logger.info(f"📊 Monitor 명령어 완료 - 주제: {주제}, 기간: {period_description}, 사용자: {user.name}")

    except Exception as e:
        logger.error(f"💥 monitor 명령어 실행 중 오류: {e}", exc_info=True)

        error_embed = discord.Embed(
            title="❌ 오류 발생",
            description="모니터링 중 오류가 발생했습니다.",
            color=0xff0000
        )
        error_embed.add_field(
            name="오류 내용",
            value=f"```{str(e)[:1000]}```",
            inline=False
        )

        await interaction.followup.send(embed=error_embed, ephemeral=True)


@bot.tree.command(name="help", description="봇 사용법을 안내합니다")
async def help_command(interaction: discord.Interaction):
    """도움말 명령어"""
    user = interaction.user
    guild = interaction.guild
    logger.info(f"❓ /help 명령어 실행: 사용자={user.name}#{user.discriminator}, 서버={guild.name}")

    embed = discord.Embed(
        title="🤖 이슈 모니터링 봇 사용법",
        description="특정 주제에 대한 최신 이슈를 자동으로 모니터링하고 분석합니다.",
        color=0x0099ff
    )

    embed.add_field(
        name="📋 기본 명령어",
        value="```\n/monitor 주제:<주제명> 기간:<기간>\n/help - 이 도움말\n```",
        inline=False
    )

    embed.add_field(
        name="📅 기간 형식 예시",
        value="• `1일`, `3일`, `7일`\n• `1주일`, `2주일`\n• `1개월`, `3개월`\n• `24시간`, `72시간`",
        inline=True
    )

    embed.add_field(
        name="🔍 주제 예시",
        value="• `AI 기술 발전`\n• `암호화폐 시장`\n• `기후변화 대응`\n• `전기차 산업`",
        inline=True
    )

    embed.add_field(
        name="⚡ 기능",
        value="• LLM 기반 키워드 자동 생성\n• 실시간 이슈 검색\n• 신뢰도 검증\n• 구조화된 보고서 생성",
        inline=False
    )

    embed.set_footer(text="문의사항이 있으시면 개발자에게 연락해주세요")

    await interaction.response.send_message(embed=embed)
    logger.info(f"📤 도움말 응답 전송 완료 (사용자: {user.name})")


@bot.event
async def on_command_error(ctx, error):
    """명령어 오류 처리"""
    logger.error(f"Command error: {error}")

    if isinstance(error, commands.CommandNotFound):
        return  # 슬래시 명령어 사용을 권장하므로 무시

    await ctx.send(f"❌ 오류가 발생했습니다: {error}")


def run_bot():
    """봇 실행 함수"""
    try:
        # 설정 로드
        logger.info("🔧 설정 로딩 중...")
        config = Config()

        if not config.DISCORD_BOT_TOKEN:
            logger.critical("❌ Discord 봇 토큰이 환경변수에 없습니다. .env 파일을 확인해주세요!")
            logger.info("💡 .env 파일에 DISCORD_BOT_TOKEN=your_token_here 를 추가해주세요")
            return

        # 토큰 일부만 로그에 출력 (보안)
        token_preview = config.DISCORD_BOT_TOKEN[:10] + "..." if len(config.DISCORD_BOT_TOKEN) > 10 else "짧은토큰"
        logger.info(f"🔑 Discord 토큰 로드됨: {token_preview}")

        logger.info("🚀 Discord 봇 시작 중...")
        bot.run(config.DISCORD_BOT_TOKEN, log_handler=None)  # Discord.py 로그 비활성화

    except KeyboardInterrupt:
        logger.info("🛑 사용자가 봇 종료를 요청했습니다 (Ctrl+C)")
    except Exception as e:
        logger.critical(f"💥 봇 시작 실패: {e}", exc_info=True)


if __name__ == "__main__":
    run_bot()