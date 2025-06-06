import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
import tempfile
from loguru import logger
from config import config

# 키워드 생성기 import
try:
    from keyword_generator import create_keyword_generator, generate_keywords_for_topic
    KEYWORD_GENERATION_AVAILABLE = True
    logger.info("✅ 키워드 생성 모듈 로드 완료")
except ImportError as e:
    KEYWORD_GENERATION_AVAILABLE = False
    logger.warning(f"⚠️ 키워드 생성 모듈 로드 실패: {e}")

# 이슈 검색기 import
try:
    from issue_searcher import (
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
    os.remove(log_file)

logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    encoding="utf-8"
)

# 에러 로그
error_log_file = "logs/error.log"
if os.path.exists(error_log_file):
    os.remove(error_log_file)

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
        # 입력값 검증
        if not validate_topic(주제):
            logger.warning(f"❌ 잘못된 주제 입력: '{주제}' (사용자: {user.name})")
            await interaction.followup.send(
                "❌ 주제를 2글자 이상 입력해주세요.",
                ephemeral=True
            )
            return

        start_date, period_description = parse_time_period(기간)
        logger.info(f"✅ 입력값 검증 완료: 주제='{주제}', 기간='{period_description}', 세부분석={세부분석}")

        available_stage = config.get_current_stage()

        # 초기 응답
        embed = discord.Embed(
            title="🔍 이슈 모니터링 시작",
            description=f"**주제**: {주제}\n**기간**: {period_description}\n**세부분석**: {'활성화' if 세부분석 else '비활성화'}\n**구현 단계**: {available_stage}단계",
            color=0x00ff00,
            timestamp=datetime.now()
        )

        if available_stage >= 4:
            if 세부분석:
                progress_text = "```\n⏳ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 세부 정보 수집 대기\n⬜ 보고서 생성 대기\n```"
            else:
                progress_text = "```\n⏳ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⚠️ 세부 정보 수집 건너뜀\n⬜ 보고서 생성 대기\n```"
        elif available_stage >= 3:
            progress_text = "```\n⏳ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 세부 정보 수집 대기\n⬜ 보고서 생성 대기\n```"
        elif available_stage >= 2:
            progress_text = "```\n⏳ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 세부 정보 수집 대기\n⬜ 보고서 생성 대기\n```"
        else:
            progress_text = "```\n⚠️ 키워드 생성 기능 준비 중...\n⬜ 이슈 검색 미구현\n⬜ 세부 정보 수집 미구현\n⬜ 보고서 생성 미구현\n```"

        embed.add_field(
            name="📊 진행 상황",
            value=progress_text,
            inline=False
        )

        if available_stage >= 4 and 세부분석:
            embed.set_footer(text="예상 소요 시간: 2-5분 (세부 분석 포함)")
        elif available_stage >= 3:
            embed.set_footer(text="예상 소요 시간: 1-3분")
        else:
            embed.set_footer(text="설정 완료 후 기능 사용이 가능합니다")

        await interaction.followup.send(embed=embed)
        logger.info(f"📤 초기 응답 전송 완료 (사용자: {user.name})")

        # 단계별 처리
        if available_stage >= 2 and KEYWORD_GENERATION_AVAILABLE:
            try:
                logger.info(f"1단계 시작: 키워드 생성 (주제: {주제})")

                # 진행상황 업데이트
                embed.set_field_at(0,
                                   name="📊 진행 상황",
                                   value="```\n✅ 키워드 생성 중...\n⬜ 이슈 검색 대기\n⬜ 세부 정보 수집 대기\n⬜ 보고서 생성 대기\n```",
                                   inline=False
                                   )
                await interaction.edit_original_response(embed=embed)

                # 키워드 생성 실행
                keyword_result = await generate_keywords_for_topic(주제)
                logger.success(f"키워드 생성 완료: {len(keyword_result.primary_keywords)}개 핵심 키워드")

                # 이슈 검색 (3단계 이상에서 실행)
                if available_stage >= 3 and ISSUE_SEARCH_AVAILABLE:
                    # 진행상황 업데이트
                    embed.set_field_at(0,
                                       name="📊 진행 상황",
                                       value="```\n✅ 키워드 생성 완료\n⏳ 이슈 검색 중...\n⬜ 세부 정보 수집 대기\n⬜ 보고서 생성 대기\n```",
                                       inline=False
                                       )
                    await interaction.edit_original_response(embed=embed)

                    logger.info(f"3단계 시작: 이슈 검색 (세부분석: {세부분석})")

                    # 이슈 검색 실행
                    search_result = await search_issues_for_keywords(
                        keyword_result,
                        period_description,
                        collect_details=세부분석 and available_stage >= 3
                    )

                    logger.success(
                        f"이슈 검색 완료: {search_result.total_found}개 이슈, 세부분석 {search_result.detailed_issues_count}개")

                    # 세부 정보 수집 상태 업데이트
                    if 세부분석 and available_stage >= 3:
                        embed.set_field_at(0,
                                           name="📊 진행 상황",
                                           value="```\n✅ 키워드 생성 완료\n✅ 이슈 검색 완료\n✅ 세부 정보 수집 완료\n⏳ 보고서 생성 중...\n```",
                                           inline=False
                                           )
                    else:
                        embed.set_field_at(0,
                                           name="📊 진행 상황",
                                           value="```\n✅ 키워드 생성 완료\n✅ 이슈 검색 완료\n⚠️ 세부 정보 수집 건너뜀\n⏳ 보고서 생성 중...\n```",
                                           inline=False
                                           )

                    # 키워드 결과 추가
                    keyword_summary = create_keyword_generator().format_keywords_summary(keyword_result)
                    embed.add_field(
                        name="🎯 생성된 키워드",
                        value=keyword_summary,
                        inline=False
                    )

                    # 이슈 검색 결과 추가
                    search_summary = create_issue_searcher().format_search_summary(search_result)
                    embed.add_field(
                        name="🔍 검색 결과",
                        value=search_summary[:1024],
                        inline=False
                    )

                    await interaction.edit_original_response(embed=embed)

                    # 최종 완료 메시지
                    stage_text = "4단계" if (세부분석 and available_stage >= 4) else "3단계"
                    success_embed = discord.Embed(
                        title=f"✅ {stage_text} 완료: 이슈 모니터링",
                        description=f"주제 '{주제}'에 대한 이슈 모니터링이 완료되었습니다.",
                        color=0x00ff00
                    )

                    # 결과 요약
                    result_summary = f"📊 **검색 결과**\n"
                    result_summary += f"• 총 {search_result.total_found}개 이슈 발견\n"
                    result_summary += f"• 검색 신뢰도: {int(search_result.confidence_score * 100)}%\n"
                    result_summary += f"• 소요 시간: {search_result.search_time:.1f}초\n"

                    if search_result.detailed_issues_count > 0:
                        result_summary += f"• 세부 분석: {search_result.detailed_issues_count}개 이슈\n"
                        result_summary += f"• 세부 신뢰도: {int(search_result.average_detail_confidence * 100)}%\n"
                        result_summary += f"• 세부 분석 시간: {search_result.total_detail_collection_time:.1f}초\n"

                    success_embed.add_field(
                        name="📈 분석 결과",
                        value=result_summary,
                        inline=False
                    )

                    # 상위 이슈 미리보기
                    if search_result.issues:
                        preview_text = ""
                        for i, issue in enumerate(search_result.issues[:3], 1):
                            preview_text += f"**{i}. {issue.title[:50]}{'...' if len(issue.title) > 50 else ''}**\n"
                            preview_text += f"📰 {issue.source} | 관련도: {int(issue.relevance_score * 100)}%"

                            if issue.detail_confidence and issue.detail_confidence > 0:
                                preview_text += f" | 세부: {int(issue.detail_confidence * 100)}%"

                            preview_text += "\n"

                            # 영향도 표시
                            if issue.impact_analysis:
                                impact_emoji = {
                                    "low": "🟢", "medium": "🟡",
                                    "high": "🟠", "critical": "🔴"
                                }.get(issue.impact_analysis.impact_level, "⚪")
                                preview_text += f"{impact_emoji} 영향도: {issue.impact_analysis.impact_level}"
                                if issue.impact_analysis.affected_sectors:
                                    preview_text += f" ({', '.join(issue.impact_analysis.affected_sectors[:2])})"
                                preview_text += "\n"

                            # 관련 인물/기관 표시
                            if issue.related_entities:
                                top_entities = [e.name for e in
                                                sorted(issue.related_entities, key=lambda x: x.relevance, reverse=True)[
                                                :2]]
                                preview_text += f"👥 관련: {', '.join(top_entities)}\n"

                            preview_text += "\n"

                        success_embed.add_field(
                            name="🔝 주요 이슈 미리보기",
                            value=preview_text[:1024],
                            inline=False
                        )

                    # 상세 보고서 생성 및 파일 첨부
                    if search_result.detailed_issues_count > 0:
                        try:
                            detailed_report = create_detailed_report_from_search_result(search_result)

                            # 임시 파일로 저장
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False,
                                                             encoding='utf-8') as f:
                                f.write(detailed_report)
                                temp_file_path = f.name

                            filename = f"issue_report_{주제.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

                            success_embed.add_field(
                                name="📋 상세 보고서",
                                value=f"세부 분석된 {search_result.detailed_issues_count}개 이슈에 대한 상세 보고서가 첨부되었습니다.",
                                inline=False
                            )

                            # 파일과 함께 전송
                            with open(temp_file_path, 'rb') as f:
                                discord_file = discord.File(f, filename=filename)
                                await interaction.followup.send(embed=success_embed, file=discord_file)

                            # 임시 파일 정리
                            os.unlink(temp_file_path)

                            logger.success(f"상세 보고서 첨부 완료: {filename}")

                        except Exception as e:
                            logger.error(f"상세 보고서 생성 실패: {e}")
                            await interaction.followup.send(embed=success_embed)
                    else:
                        await interaction.followup.send(embed=success_embed)

                else:
                    # 3단계 미지원 안내
                    embed.set_field_at(0,
                                       name="📊 진행 상황",
                                       value="```\n✅ 키워드 생성 완료\n⚠️ 이슈 검색 기능 준비 중\n⬜ 세부 정보 수집 대기\n⬜ 보고서 생성 대기\n```",
                                       inline=False
                                       )

                    # 키워드 결과만 표시
                    keyword_summary = create_keyword_generator().format_keywords_summary(keyword_result)
                    embed.add_field(
                        name="🎯 생성된 키워드",
                        value=keyword_summary,
                        inline=False
                    )

                    await interaction.edit_original_response(embed=embed)

                    limitation_embed = discord.Embed(
                        title="⚠️ 기능 제한 (2단계까지 완료)",
                        description="키워드 생성은 완료되었으나, 이슈 검색을 위해 추가 설정이 필요합니다.",
                        color=0xffaa00
                    )

                    limitation_embed.add_field(
                        name="💡 다음 단계 진행 방법",
                        value="`.env` 파일에 `PERPLEXITY_API_KEY=your_key_here`를 추가하고 봇을 재시작하세요.",
                        inline=False
                    )

                    limitation_embed.add_field(
                        name="🎯 생성된 키워드 활용",
                        value=f"총 {len(create_keyword_generator().get_all_keywords(keyword_result))}개 키워드가 생성되어 향후 검색에 사용됩니다.",
                        inline=False
                    )

                    await interaction.followup.send(embed=limitation_embed)

            except Exception as e:
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

            setup_guide += "🚀 **4단계**: 세부 정보 수집 기능 구현 완료\n"

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
            elif available_stage == 2:
                limitation_embed.add_field(
                    name="💡 다음 단계 진행 방법",
                    value="`.env` 파일에 `PERPLEXITY_API_KEY=your_key_here`를 추가하고 봇을 재시작하세요.",
                    inline=False
                )

            # 현재 구현된 기능 상태
            feature_status = f"• 입력값 검증: ✅\n• 시간 파싱: ✅\n"
            feature_status += f"• 키워드 생성: {'✅' if available_stage >= 2 else '⏳'}\n"
            feature_status += f"• 이슈 검색: {'✅' if available_stage >= 3 else '⏳'}\n"
            feature_status += f"• 세부 정보 수집: {'✅' if available_stage >= 4 else '⏳'}\n"
            feature_status += f"• 환각 탐지: ⏳ (5단계 예정)\n"
            feature_status += f"• 고급 보고서: ⏳ (6단계 예정)"

            limitation_embed.add_field(
                name="🚧 현재 구현 상태",
                value=feature_status,
                inline=False
            )

            await interaction.followup.send(embed=limitation_embed)

        logger.info(
            f"📊 Monitor 명령어 완료 - 주제: {주제}, 기간: {period_description}, 세부분석: {세부분석}, 사용자: {user.name}, 단계: {available_stage}")

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
    logger.info(f"❓ /help 명령어 실행: 사용자={user.name}, 서버={guild.name}")

    current_stage = config.get_current_stage()

    embed = discord.Embed(
        title="🤖 이슈 모니터링 봇 사용법",
        description=f"특정 주제에 대한 최신 이슈를 자동으로 모니터링하고 분석합니다.\n**현재 구현 단계**: {current_stage}단계",
        color=0x0099ff
    )

    embed.add_field(
        name="📋 기본 명령어",
        value="```\n/monitor 주제:<주제명> 기간:<기간> 세부분석:<True/False>\n/help - 이 도움말\n/status - 시스템 상태 확인\n```",
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
        stage_features += "✅ **3단계**: Perplexity API 실시간 이슈 검색\n"
    if current_stage >= 4:
        stage_features += "✅ **4단계**: 세부 정보 수집 및 분석\n"
    else:
        stage_features += "⏳ **4단계**: 세부 정보 수집 (구현 완료)\n"

    stage_features += "⏳ **5단계**: 신뢰도 검증 (예정)\n"
    stage_features += "⏳ **6단계**: 구조화된 보고서 생성 (예정)"

    embed.add_field(
        name="⚡ 단계별 기능",
        value=stage_features,
        inline=False
    )

    # 4단계 세부 기능 안내
    if current_stage >= 4:
        detail_features = ""
        detail_features += "🔍 **관련 인물/기관**: 이슈 관련 핵심 인물과 기관 추출\n"
        detail_features += "📊 **영향도 분석**: 파급효과 및 지리적 범위 평가\n"
        detail_features += "⏰ **시간순 전개**: 이슈 발전 과정 추적\n"
        detail_features += "📋 **상세 보고서**: 마크다운 파일 자동 생성"

        embed.add_field(
            name="🚀 4단계 세부 기능",
            value=detail_features,
            inline=False
        )

    # 사용 예시
    if current_stage >= 4:
        embed.add_field(
            name="💡 사용 예시",
            value="```\n/monitor 주제:AI 기술 발전 기간:1주일 세부분석:True\n```\n"
                  "→ AI 관련 키워드 자동 생성\n"
                  "→ 최근 1주일 이슈 검색\n"
                  "→ 관련 인물/기관 및 영향도 분석\n"
                  "→ 상세 보고서 파일 제공",
            inline=False
        )
    elif current_stage >= 3:
        embed.add_field(
            name="💡 사용 예시",
            value="```\n/monitor 주제:AI 기술 발전 기간:1주일\n```\n"
                  "→ AI 관련 키워드 자동 생성\n"
                  "→ 최근 1주일 이슈 검색\n"
                  "→ 관련성 점수로 정렬된 결과 제공",
            inline=False
        )
    elif current_stage >= 2:
        embed.add_field(
            name="💡 현재 사용 가능",
            value="```\n/monitor 주제:AI 기술 발전 기간:1주일\n```\n"
                  "→ AI 관련 키워드 자동 생성\n"
                  "→ 이슈 검색은 Perplexity API 키 설정 후 사용 가능",
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
    logger.info(f"📊 /status 명령어 실행: 사용자={user.name}, 서버={guild.name}")

    stage_info = config.get_stage_info()
    current_stage = config.get_current_stage()

    embed = discord.Embed(
        title="📊 시스템 상태",
        description=f"현재 실행 가능한 최고 단계: **{current_stage}단계**",
        color=0x00ff00 if current_stage >= 4 else (
            0x3498db if current_stage >= 3 else (0xffaa00 if current_stage >= 2 else 0xff0000)),
        timestamp=datetime.now()
    )

    # 단계별 상태
    status_text = ""
    status_text += f"{'✅' if stage_info['stage1_discord'] else '❌'} **1단계**: Discord 봇 연결\n"
    status_text += f"{'✅' if stage_info['stage2_openai'] else '❌'} **2단계**: 키워드 생성 (OpenAI)\n"
    status_text += f"{'✅' if stage_info['stage3_perplexity'] else '❌'} **3단계**: 이슈 검색 (Perplexity)\n"
    status_text += f"{'✅' if current_stage >= 4 else '⏳'} **4단계**: 세부 정보 수집\n"
    status_text += f"⏳ **5단계**: 환각 탐지 및 검증 (예정)\n"
    status_text += f"⏳ **6단계**: 구조화된 보고서 생성 (예정)"

    embed.add_field(
        name="🔧 단계별 준비 상태",
        value=status_text,
        inline=False
    )

    # 4단계 기능 상세 정보
    if current_stage >= 4:
        detail_features = ""
        detail_features += "🔍 **세부 정보 수집**: 관련 인물/기관 추출\n"
        detail_features += "📊 **영향도 분석**: 파급효과 및 중요도 평가\n"
        detail_features += "⏰ **시간순 전개**: 이슈 발전 과정 추적\n"
        detail_features += "🎯 **신뢰도 계산**: 세부 정보 품질 평가\n"
        detail_features += "📋 **상세 보고서**: 마크다운 파일 자동 생성"

        embed.add_field(
            name="🚀 4단계 세부 기능",
            value=detail_features,
            inline=False
        )

    # 모듈 상태
    module_status = ""
    module_status += f"✅ Discord.py: 연결됨\n"
    module_status += f"{'✅' if KEYWORD_GENERATION_AVAILABLE else '❌'} 키워드 생성: {'사용 가능' if KEYWORD_GENERATION_AVAILABLE else '설정 필요'}\n"
    module_status += f"{'✅' if ISSUE_SEARCH_AVAILABLE else '❌'} 이슈 검색: {'사용 가능' if ISSUE_SEARCH_AVAILABLE else '설정 필요'}\n"
    module_status += f"{'✅' if current_stage >= 4 else '⏳'} 세부 정보 수집: {'사용 가능' if current_stage >= 4 else '준비 중'}\n"
    module_status += f"⏳ 환각 탐지: 준비 중 (5단계)\n"
    module_status += f"⏳ 고급 보고서: 준비 중 (6단계)"

    embed.add_field(
        name="📦 모듈 상태",
        value=module_status,
        inline=True
    )

    # 설정 정보
    config_text = ""
    config_text += f"개발 모드: {'ON' if stage_info['development_mode'] else 'OFF'}\n"
    config_text += f"로그 레벨: {stage_info['log_level']}\n"
    config_text += f"서버 수: {len(bot.guilds)}개\n"
    config_text += f"4단계 지원: {'✅' if current_stage >= 4 else '❌'}"

    embed.add_field(
        name="⚙️ 설정 정보",
        value=config_text,
        inline=True
    )

    # 다음 단계 안내
    if current_stage < 6:
        next_step = ""
        if current_stage < 1:
            next_step = "Discord 봇 토큰을 .env 파일에 추가하여 봇을 활성화하세요."
        elif current_stage < 2:
            next_step = "OpenAI API 키를 .env 파일에 추가하여 키워드 생성 기능을 활성화하세요."
        elif current_stage < 3:
            next_step = "Perplexity API 키를 .env 파일에 추가하여 이슈 검색 기능을 활성화하세요."
        elif current_stage < 4:
            next_step = "4단계 세부 정보 수집 기능이 구현되었습니다! 모든 API 키가 설정되면 사용 가능합니다."
        else:
            next_step = "현재 4단계까지 완전 구현되었습니다. 5-6단계는 개발 예정입니다."

        embed.add_field(
            name="💡 다음 단계",
            value=next_step,
            inline=False
        )

    # API 키 상태 (마스킹)
    api_status = ""
    discord_token = config.get_discord_token()
    openai_key = config.get_openai_api_key()
    perplexity_key = config.get_perplexity_api_key()

    if discord_token:
        api_status += f"🔑 Discord: {discord_token[:8]}...{discord_token[-4:] if len(discord_token) > 12 else '***'}\n"
    else:
        api_status += f"❌ Discord: 설정되지 않음\n"

    if openai_key:
        api_status += f"🔑 OpenAI: {openai_key[:8]}...{openai_key[-4:] if len(openai_key) > 12 else '***'}\n"
    else:
        api_status += f"❌ OpenAI: 설정되지 않음\n"

    if perplexity_key:
        api_status += f"🔑 Perplexity: {perplexity_key[:8]}...{perplexity_key[-4:] if len(perplexity_key) > 12 else '***'}\n"
    else:
        api_status += f"❌ Perplexity: 설정되지 않음\n"

    embed.add_field(
        name="🔐 API 키 상태",
        value=api_status,
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


def check_module_availability():
    """모듈 가용성 확인 및 로깅"""
    modules_status = {
        "discord.py": True,
        "keyword_generation": KEYWORD_GENERATION_AVAILABLE,
        "issue_search": ISSUE_SEARCH_AVAILABLE,
    }

    logger.info("📦 모듈 가용성 확인:")
    for module, available in modules_status.items():
        status = "✅ 사용 가능" if available else "❌ 사용 불가"
        logger.info(f"  {module}: {status}")

    # 권장사항 출력
    if not KEYWORD_GENERATION_AVAILABLE:
        logger.info("💡 키워드 생성 기능 활성화: pip install openai && OPENAI_API_KEY 설정")

    if not ISSUE_SEARCH_AVAILABLE:
        logger.info("💡 이슈 검색 기능 활성화: pip install httpx && PERPLEXITY_API_KEY 설정")

    return modules_status


def run_bot():
    """봇 실행 함수"""
    try:
        logger.info("🔧 설정 로딩 중...")

        # 모듈 가용성 확인
        modules_status = check_module_availability()

        # 기본 설정 상태 출력
        if config.is_development_mode():
            config.print_stage_status()

        # Discord 토큰 확인
        discord_token = config.get_discord_token()
        if not discord_token:
            logger.critical("❌ Discord 봇 토큰이 환경변수에 없습니다. .env 파일을 확인해주세요!")
            logger.info("💡 .env 파일에 DISCORD_BOT_TOKEN=your_token_here 를 추가해주세요")
            return

        token_preview = discord_token[:10] + "..." if len(discord_token) > 10 else "짧은토큰"
        logger.info(f"🔑 Discord 토큰 로드됨: {token_preview}")

        # 키워드 생성 기능 상태 확인
        if config.validate_stage2_requirements() and KEYWORD_GENERATION_AVAILABLE:
            logger.success("✅ 키워드 생성 기능 사용 가능")
        else:
            logger.warning("⚠️ 키워드 생성 기능 사용 불가")
            if not config.get_openai_api_key():
                logger.info("💡 OpenAI API 키 설정 필요")
            if not KEYWORD_GENERATION_AVAILABLE:
                logger.info("💡 OpenAI 패키지 설치 필요: pip install openai")

        # 이슈 검색 기능 상태 확인
        if config.validate_stage3_requirements() and ISSUE_SEARCH_AVAILABLE:
            logger.success("✅ 이슈 검색 기능 사용 가능")
        else:
            logger.info("⚠️ 이슈 검색 기능 사용 불가")
            if not config.get_perplexity_api_key():
                logger.info("💡 Perplexity API 키 설정 필요")
            if not ISSUE_SEARCH_AVAILABLE:
                logger.info("💡 httpx 패키지 설치 필요: pip install httpx")

        # 최종 실행 가능 단계 확인
        final_stage = config.get_current_stage()
        if KEYWORD_GENERATION_AVAILABLE and final_stage >= 2:
            if ISSUE_SEARCH_AVAILABLE and final_stage >= 3:
                logger.success(f"🚀 {final_stage}단계까지 모든 기능 사용 가능 (4단계 세부 분석 포함)")
            else:
                logger.info(f"⚡ {final_stage}단계까지 사용 가능 (이슈 검색 제외)")
        else:
            logger.info(f"⚡ {final_stage}단계까지 사용 가능")

        # 4단계 기능 안내
        if final_stage >= 4:
            logger.success("🔍 4단계 세부 정보 수집 기능 구현 완료!")
            logger.info("   • 관련 인물/기관 추출")
            logger.info("   • 영향도 분석 및 평가")
            logger.info("   • 시간순 이벤트 추적")
            logger.info("   • 상세 보고서 자동 생성")

        logger.info("🚀 Discord 봇 시작 중...")
        bot.run(discord_token, log_handler=None)

    except KeyboardInterrupt:
        logger.info("🛑 사용자가 봇 종료를 요청했습니다 (Ctrl+C)")
    except Exception as e:
        logger.critical(f"💥 봇 시작 실패: {e}", exc_info=True)


if __name__ == "__main__":
    run_bot()