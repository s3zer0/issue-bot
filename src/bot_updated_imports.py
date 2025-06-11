"""
Discord 봇의 메인 진입점 (Clean Architecture 적용 버전).

새로운 Clean Architecture 구조를 사용하는 업데이트된 Discord 봇입니다.
순환 의존성이 제거되고 관심사의 분리가 명확하게 이루어졌습니다.

주요 변경사항:
- 기존 순환 의존성 구조에서 Clean Architecture로 전환
- 의존성 주입(DI) 컨테이너 사용
- 프레젠테이션 레이어와 비즈니스 로직 분리
- 새로운 import 경로 적용
"""

import discord
from discord.ext import commands
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger

# --- Clean Architecture 임포트 ---
# Dependency Injection
from src_new.infrastructure.container.dependency_injection import container

# Use Cases (Application Layer)
from src_new.application.use_cases.analyze_issues import AnalyzeIssuesUseCase

# DTOs (Application Layer)
from src_new.application.dto.issue_requests import AnalyzeIssuesRequest
from src_new.application.dto.issue_responses import AnalyzeIssuesResponse

# Value Objects (Domain Layer)
from src_new.domain.value_objects.time_period import TimePeriod
from src_new.domain.value_objects.confidence import Confidence

# Presentation Layer
from src_new.presentation.discord.commands.analyze_command import AnalyzeCommand
from src_new.presentation.discord.formatters.discord_formatter import DiscordFormatter

# --- 로깅 설정 ---
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO", colorize=True)
logger.add("logs/bot.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO", encoding="utf-8")
logger.add("logs/error.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="ERROR", encoding="utf-8")

# --- Clean Architecture 봇 클래스 ---
intents = discord.Intents.default()
intents.message_content = True

class IssueMonitorBotClean(commands.Bot):
    """Clean Architecture를 적용한 Discord 봇."""
    
    def __init__(self):
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        
        # 의존성 주입 컨테이너 설정
        self._setup_dependencies()
        
        # 프레젠테이션 레이어 컴포넌트 초기화
        self.formatter = DiscordFormatter()
        self.analyze_command = AnalyzeCommand(
            use_case=container.get(AnalyzeIssuesUseCase),
            formatter=self.formatter
        )
        
        logger.info("🤖 Clean Architecture IssueMonitorBot 인스턴스 생성됨")

    def _setup_dependencies(self):
        """의존성 주입 컨테이너 설정"""
        try:
            container.configure_dependencies()
            logger.info("✅ 의존성 주입 컨테이너 설정 완료")
        except Exception as e:
            logger.error(f"❌ 의존성 설정 실패: {e}")
            raise

    async def setup_hook(self):
        """봇이 Discord에 로그인한 후 실행되는 설정 메서드."""
        logger.info("⚙️ Clean Architecture 봇 셋업 시작...")
        try:
            # 슬래시 명령어 동기화
            synced = await self.tree.sync()
            logger.info(f"✅ {len(synced)}개의 슬래시 명령어 동기화 완료")
        except Exception as e:
            logger.error(f"❌ 명령어 동기화 실패: {e}")

    async def on_ready(self):
        """봇이 준비되었을 때 호출되는 이벤트 핸들러."""
        logger.info(f"✅ {self.user}로 로그인했습니다! (Clean Architecture)")
        logger.info(f"📊 {len(self.guilds)}개의 서버에 연결됨")

# --- 슬래시 명령어 (Clean Architecture 적용) ---
bot = IssueMonitorBotClean()

@bot.tree.command(name="analyze", description="AI 기반 이슈 분석 및 환각 탐지")
@discord.app_commands.describe(
    topic="분석할 주제나 키워드",
    time_period="검색 기간 (예: 최근 1주일, 최근 1개월)"
)
async def analyze_issues_clean(
    interaction: discord.Interaction, 
    topic: str, 
    time_period: str = "최근 1주일"
):
    """Clean Architecture를 사용한 이슈 분석 명령어."""
    try:
        # 명령어 실행을 프레젠테이션 레이어에 위임
        await bot.analyze_command.execute_for_interaction(interaction, topic, time_period)
        
    except Exception as e:
        logger.error(f"❌ 분석 명령어 실행 중 오류: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message(
                "❌ 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                ephemeral=True
            )

@bot.tree.command(name="status", description="봇 상태 및 시스템 정보 확인")
async def status_clean(interaction: discord.Interaction):
    """Clean Architecture 봇의 상태 확인."""
    try:
        # 의존성 컨테이너에서 사용 가능한 서비스 확인
        available_detectors = len(container.get_all_detectors())
        
        embed = bot.formatter.create_status_embed(
            is_ready=True,
            available_detectors=available_detectors
        )
        
        embed.add_field(
            name="🏗️ 아키텍처",
            value="Clean Architecture 적용",
            inline=True
        )
        
        embed.add_field(
            name="🔧 의존성 주입",
            value="활성화됨",
            inline=True
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"❌ 상태 확인 중 오류: {e}")
        await interaction.response.send_message(
            "❌ 상태 확인 중 오류가 발생했습니다.",
            ephemeral=True
        )

@bot.tree.command(name="help", description="봇 사용법 및 명령어 도움말")
async def help_clean(interaction: discord.Interaction):
    """Clean Architecture 봇의 도움말."""
    try:
        embed = bot.formatter.create_help_embed()
        
        # Clean Architecture 관련 추가 정보
        embed.add_field(
            name="🆕 새로운 기능",
            value="• Clean Architecture 적용\n"
                  "• 향상된 안정성 및 성능\n"
                  "• 모듈화된 구조",
            inline=False
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"❌ 도움말 표시 중 오류: {e}")
        await interaction.response.send_message(
            "❌ 도움말을 표시할 수 없습니다.",
            ephemeral=True
        )

# --- 기존 명령어와의 호환성 유지 ---
@bot.command(name='analyze_legacy')
async def analyze_legacy_command(ctx, *, args: str = ""):
    """기존 명령어와의 호환성을 위한 레거시 명령어."""
    # 인자 파싱
    parts = args.split() if args else []
    topic = parts[0] if parts else "AI"
    time_period = " ".join(parts[1:]) if len(parts) > 1 else "최근 1주일"
    
    try:
        # 새로운 시스템으로 처리
        await bot.analyze_command.execute(ctx, topic, time_period)
        
    except Exception as e:
        logger.error(f"❌ 레거시 명령어 실행 중 오류: {e}")
        await ctx.send("❌ 분석 중 오류가 발생했습니다. 슬래시 명령어 `/analyze`를 사용해보세요.")

# --- 에러 핸들링 ---
@bot.event
async def on_command_error(ctx, error):
    """명령어 오류 처리."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ 알 수 없는 명령어입니다. `/help`를 사용해 도움말을 확인하세요.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ 필수 인자가 누락되었습니다: {error.param}")
    else:
        logger.error(f"예상치 못한 명령어 오류: {error}")
        await ctx.send("❌ 명령어 처리 중 오류가 발생했습니다.")

@bot.event
async def on_error(event, *args, **kwargs):
    """일반적인 오류 처리."""
    logger.error(f"봇 오류 발생 - 이벤트: {event}, 인자: {args}")

# --- 봇 실행 ---
async def main():
    """Clean Architecture 봇 실행."""
    try:
        # 환경 변수에서 토큰 가져오기 (기존과 동일)
        token = os.getenv('DISCORD_TOKEN')
        if not token:
            logger.error("❌ DISCORD_TOKEN 환경 변수가 설정되지 않았습니다")
            return

        logger.info("🚀 Clean Architecture 봇 시작...")
        await bot.start(token)
        
    except Exception as e:
        logger.error(f"❌ 봇 실행 실패: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())