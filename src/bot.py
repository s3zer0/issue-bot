import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger
from src.config import config

# 키워드 생성기 import
try:
    from src.keyword_generator import create_keyword_generator

    KEYWORD_GENERATION_AVAILABLE = True
    logger.info("✅ 키워드 생성 모듈 로드 완료")
except ImportError as e:
    KEYWORD_GENERATION_AVAILABLE = False
    logger.warning(f"⚠️ 키워드 생성 모듈 로드 실패: {e}")
    logger.info("💡 OpenAI 패키지를 설치하고 API 키를 설정해주세요")

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
        status_message = f"/monitor commands (Stage {current_stage})"
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=status_message
            )
        )
        logger.info(f"👀 봇 상태 설정: {status_message}")

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

        # 현재 구현 가능한 단계 확인
        available_stage = config.get_current_stage()

        # 초기 응답 (임베드로 정보 정리)
        embed = discord.Embed(
            title="🔍 이슈 모니터링 시작",
            description=f"**주제**: {주제}\n**기간**: {period_description}\n**구현 단계**: {available_stage}단계",
            color=0x00ff00,
            timestamp=datetime.now()
        )

        if available_stage >= 2:
            progress_text = "```\n⏳ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 상세 정보 수집 대기\n⬜ 보고서 생성 대기\n```"
        else:
            progress_text = "```\n⚠️ 키워드 생성 기능 준비 중...\n⬜ 이슈 검색 미구현\n⬜ 상세 정보 수집 미구현\n⬜ 보고서 생성 미구현\n```"

        embed.add_field(
            name="📊 진행 상황",
            value=progress_text,
            inline=False
        )

        if available_stage >= 2:
            embed.set_footer(text="예상 소요 시간: 1-3분")
        else:
            embed.set_footer(text="설정 완료 후 키워드 생성이 가능합니다")

        await interaction.followup.send(embed=embed)
        logger.info(f"📤 초기 응답 전송 완료 (사용자: {user.name})")

        # 단계별 처리
        if available_stage >= 2 and KEYWORD_GENERATION_AVAILABLE:
            # 1단계: LLM 키워드 생성
            try:
                logger.info(f"1단계 시작: 키워드 생성 (주제: {주제})")

                # 진행상황 업데이트
                embed.set_field_at(0,
                                   name="📊 진행 상황",
                                   value="```\n✅ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 상세 정보 수집 대기\n⬜ 보고서 생성 대기\n```",
                                   inline=False
                                   )
                await interaction.edit_original_response(embed=embed)

                # 키워드 생성 실행
                keyword_result = await generate_keywords_for_topic(주제)

                logger.success(f"키워드 생성 완료: {len(keyword_result.primary_keywords)}개 핵심 키워드")

                # 완료 상태 업데이트
                embed.set_field_at(0,
                                   name="📊 진행 상황",
                                   value="```\n✅ 키워드 생성 완료\n⏳ 이슈 검색 준비 중...\n⬜ 상세 정보 수집 대기\n⬜ 보고서 생성 대기\n```",
                                   inline=False
                                   )

                # 키워드 결과 추가
                keyword_summary = keyword_generator.format_keywords_summary(keyword_result)

                embed.add_field(
                    name="🎯 생성된 키워드",
                    value=keyword_summary,
                    inline=False
                )

                await interaction.edit_original_response(embed=embed)

                # TODO: 2단계 - Perplexity API 이슈 탐색 (다음 구현)
                # TODO: 3단계 - 세부 정보 수집 (다음 구현)
                # TODO: 4단계 - 환각 탐지 (다음 구현)
                # TODO: 5단계 - 보고서 생성 (다음 구현)

                # 임시 완료 메시지 (2단계 완료 후)
                logger.info("⏳ 다음 단계 대기 중 (Perplexity API 연동 예정)")
                await asyncio.sleep(1)  # 시뮬레이션

                success_embed = discord.Embed(
                    title="✅ 2단계 완료: 키워드 생성",
                    description=f"주제 '{주제}'에 대한 키워드 생성이 완료되었습니다.",
                    color=0x00ff00
                )
                success_embed.add_field(
                    name="📈 다음 단계 예정",
                    value="3단계: Perplexity API를 통한 이슈 검색\n4단계: 세부 정보 수집\n5단계: 환각 탐지 및 검증\n6단계: 보고서 생성 및 전송",
                    inline=False
                )

                total_keywords = len(keyword_generator.get_all_keywords(keyword_result))
                success_embed.add_field(
                    name="🔗 생성된 키워드 활용",
                    value=f"총 {total_keywords}개 키워드가 다음 단계 검색에 사용됩니다.\n소요시간: {keyword_result.generation_time:.1f}초",
                    inline=False
                )

                await interaction.followup.send(embed=success_embed)

            except KeywordGenerationError as e:
                logger.error(f"키워드 생성 실패: {e}")

                error_embed = discord.Embed(
                    title="❌ 키워드 생성 실패",
                    description=f"주제 '{주제}'에 대한 키워드 생성 중 오류가 발생했습니다.",
                    color=0xff0000
                )
                error_embed.add_field(
                    name="오류 내용",
                    value=f"```{str(e)[:500]}```",
                    inline=False
                )
                error_embed.add_field(
                    name="🔧 확인사항",
                    value="• OpenAI API 키가 .env 파일에 설정되어 있는지 확인\n• 인터넷 연결 상태 확인\n• API 사용량 제한 확인",
                    inline=False
                )

                await interaction.followup.send(embed=error_embed, ephemeral=True)
                return

        else:
            # 단계별 제한 안내
            logger.info(f"⚠️ 현재 단계 제한: {available_stage}단계 (사용자: {user.name})")

            limitation_embed = discord.Embed(
                title="⚠️ 기능 제한",
                description=f"현재 {available_stage}단계까지만 구현되어 있습니다.",
                color=0xffaa00
            )

            stage_info = config.get_stage_info()
            setup_guide = ""

            if not stage_info["stage1_discord"]:
                setup_guide += "❌ **1단계**: DISCORD_BOT_TOKEN 설정 필요\n"
            else:
                setup_guide += "✅ **1단계**: Discord 봇 연결 완료\n"

            if not stage_info["stage2_openai"]:
                setup_guide += "❌ **2단계**: OPENAI_API_KEY 설정 필요\n"
            else:
                setup_guide += "✅ **2단계**: 키워드 생성 기능 사용 가능\n"

            if not stage_info["stage3_perplexity"]:
                setup_guide += "⏳ **3단계**: PERPLEXITY_API_KEY 설정 시 이슈 검색 가능\n"
            else:
                setup_guide += "✅ **3단계**: 이슈 검색 기능 사용 가능\n"

            limitation_embed.add_field(
                name="🔧 설정 상태",
                value=setup_guide,
                inline=False
            )

            if available_stage == 1:
                limitation_embed.add_field(
                    name="💡 다음 단계 진행 방법",
                    value="`.env` 파일에 `OPENAI_API_KEY=your_key_here`를 추가하고 봇을 재시작하세요.",
                    inline=False
                )

            # 현재 구현된 기능만 시뮬레이션
            await asyncio.sleep(2)

            limitation_embed.add_field(
                name="🚧 현재 구현 상태",
                value=f"• 입력값 검증: ✅\n• 시간 파싱: ✅\n• 키워드 생성: {'✅' if available_stage >= 2 else '⏳'}\n• 이슈 검색: ⏳\n• 보고서 생성: ⏳",
                inline=False
            )

            await interaction.followup.send(embed=limitation_embed)

        # 로깅
        logger.info(f"📊 Monitor 명령어 완료 - 주제: {주제}, 기간: {period_description}, 사용자: {user.name}, 단계: {available_stage}")

    except Exception as e:
        logger.error(f"💥 monitor 명령어 실행 중 오류: {e}", exc_info=True)

        error_embed = discord.Embed(
            title="❌ 시스템 오류 발생",
            description="모니터링 중 예상치 못한 오류가 발생했습니다.",
            color=0xff0000
        )
        error_embed.add_field(
            name="오류 내용",
            value=f"```{str(e)[:1000]}```",
            inline=False
        )
        error_embed.add_field(
            name="🔧 문제 해결",
            value="• 봇 로그를 확인해주세요\n• 문제가 지속되면 개발자에게 문의하세요\n• `/status` 명령어로 시스템 상태를 확인해보세요",
            inline=False
        )

        await interaction.followup.send(embed=error_embed, ephemeral=True)


@bot.tree.command(name="help", description="봇 사용법을 안내합니다")
async def help_command(interaction: discord.Interaction):
    """도움말 명령어"""
    user = interaction.user
    guild = interaction.guild
    logger.info(f"❓ /help 명령어 실행: 사용자={user.name}#{user.discriminator}, 서버={guild.name}")

    current_stage = config.get_current_stage()

    embed = discord.Embed(
        title="🤖 이슈 모니터링 봇 사용법",
        description=f"특정 주제에 대한 최신 이슈를 자동으로 모니터링하고 분석합니다.\n**현재 구현 단계**: {current_stage}단계",
        color=0x0099ff
    )

    embed.add_field(
        name="📋 기본 명령어",
        value="```\n/monitor 주제:<주제명> 기간:<기간>\n/help - 이 도움말\n/status - 시스템 상태 확인\n```",
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

    # 단계별 기능 설명
    stage_features = ""
    if current_stage >= 1:
        stage_features += "✅ **1단계**: Discord 봇 기본 기능\n"
    if current_stage >= 2:
        stage_features += "✅ **2단계**: LLM 기반 키워드 자동 생성\n"
    if current_stage >= 3:
        stage_features += "✅ **3단계**: 실시간 이슈 검색\n"
    else:
        stage_features += "⏳ **3단계**: 실시간 이슈 검색 (준비 중)\n"

    stage_features += "⏳ **4단계**: 신뢰도 검증 (예정)\n"
    stage_features += "⏳ **5단계**: 구조화된 보고서 생성 (예정)"

    embed.add_field(
        name="⚡ 단계별 기능",
        value=stage_features,
        inline=False
    )

    embed.set_footer(text="문의사항이 있으시면 개발자에게 연락해주세요")

    await interaction.response.send_message(embed=embed)
    logger.info(f"📤 도움말 응답 전송 완료 (사용자: {user.name})")


@bot.tree.command(name="status", description="봇 시스템 상태를 확인합니다")
async def status_command(interaction: discord.Interaction):
    """시스템 상태 확인 명령어"""
    user = interaction.user
    guild = interaction.guild
    logger.info(f"📊 /status 명령어 실행: 사용자={user.name}#{user.discriminator}, 서버={guild.name}")

    stage_info = config.get_stage_info()
    current_stage = config.get_current_stage()

    embed = discord.Embed(
        title="📊 시스템 상태",
        description=f"현재 실행 가능한 최고 단계: **{current_stage}단계**",
        color=0x00ff00 if current_stage >= 2 else 0xffaa00,
        timestamp=datetime.now()
    )

    # 단계별 상태
    status_text = ""
    status_text += f"{'✅' if stage_info['stage1_discord'] else '❌'} **1단계**: Discord 봇 연결\n"
    status_text += f"{'✅' if stage_info['stage2_openai'] else '❌'} **2단계**: 키워드 생성 (OpenAI)\n"
    status_text += f"{'✅' if stage_info['stage3_perplexity'] else '❌'} **3단계**: 이슈 검색 (Perplexity)\n"

    embed.add_field(
        name="🔧 단계별 준비 상태",
        value=status_text,
        inline=False
    )

    # 모듈 상태
    module_status = ""
    module_status += f"✅ Discord.py: 연결됨\n"
    module_status += f"{'✅' if KEYWORD_GENERATION_AVAILABLE else '❌'} 키워드 생성: {'사용 가능' if KEYWORD_GENERATION_AVAILABLE else '설정 필요'}\n"
    module_status += f"⏳ Perplexity API: 준비 중\n"
    module_status += f"⏳ 환각 탐지: 준비 중"

    embed.add_field(
        name="📦 모듈 상태",
        value=module_status,
        inline=True
    )

    # 설정 정보
    config_text = ""
    config_text += f"개발 모드: {'ON' if stage_info['development_mode'] else 'OFF'}\n"
    config_text += f"로그 레벨: {stage_info['log_level']}\n"
    config_text += f"서버 수: {len(bot.guilds)}개"

    embed.add_field(
        name="⚙️ 설정 정보",
        value=config_text,
        inline=True
    )

    # 다음 단계 안내
    if current_stage < 3:
        next_step = ""
        if current_stage < 2:
            next_step = "OpenAI API 키를 .env 파일에 추가하여 키워드 생성 기능을 활성화하세요."
        elif current_stage < 3:
            next_step = "Perplexity API 키를 .env 파일에 추가하여 이슈 검색 기능을 활성화하세요."

        embed.add_field(
            name="💡 다음 단계",
            value=next_step,
            inline=False
        )

    embed.set_footer(text=f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    await interaction.response.send_message(embed=embed)
    logger.info(f"📤 상태 확인 응답 전송 완료 (사용자: {user.name})")


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
        # 설정 로드 및 검증
        logger.info("🔧 설정 로딩 중...")

        # 기본 설정 상태 출력
        if config.is_development_mode():
            config.print_stage_status()

        # Discord 토큰 확인 (최소 요구사항)
        discord_token = config.get_discord_token()
        if not discord_token:
            logger.critical("❌ Discord 봇 토큰이 환경변수에 없습니다. .env 파일을 확인해주세요!")
            logger.info("💡 .env 파일에 DISCORD_BOT_TOKEN=your_token_here 를 추가해주세요")
            return

        # 토큰 일부만 로그에 출력 (보안)
        token_preview = discord_token[:10] + "..." if len(discord_token) > 10 else "짧은토큰"
        logger.info(f"🔑 Discord 토큰 로드됨: {token_preview}")

        # 키워드 생성 기능 상태 확인
        if config.validate_stage2_requirements():
            logger.success("✅ 키워드 생성 기능 사용 가능")
        else:
            logger.warning("⚠️ 키워드 생성 기능 사용 불가 - OpenAI API 키 설정 필요")

        # 이슈 검색 기능 상태 확인
        if config.validate_stage3_requirements():
            logger.success("✅ 이슈 검색 기능 사용 가능")
        else:
            logger.info("💡 이슈 검색 기능 사용을 위해서는 Perplexity API 키 설정이 필요합니다")

        logger.info("🚀 Discord 봇 시작 중...")
        bot.run(discord_token, log_handler=None)  # Discord.py 로그 비활성화

    except KeyboardInterrupt:
        logger.info("🛑 사용자가 봇 종료를 요청했습니다 (Ctrl+C)")
    except Exception as e:
        logger.critical(f"💥 봇 시작 실패: {e}", exc_info=True)


if __name__ == "__main__":
    run_bot()