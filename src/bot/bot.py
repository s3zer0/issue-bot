"""
Discord 봇의 메인 진입점 (수정된 버전).

Discord API와의 상호작용, 슬래시 명령어 처리, 그리고 다른 비즈니스 로직 모듈들
(키워드 생성, 이슈 검색, 환각 탐지, 보고서 생성)의 전체 흐름을 조율(Orchestration)합니다.

주요 수정사항:
- 슬래시 명령어 매개변수명을 영어로 변경 (Discord 호환성 개선)
- 향상된 동기화 및 디버깅 기능 추가
- 봇 초대 링크 생성 명령어 추가
"""

import discord
from discord.ext import commands
from datetime import datetime, timedelta
import re
import sys
import os
import time
import asyncio
from loguru import logger

# --- 모듈 임포트 ---
from src.config import config
from src.models import KeywordResult, SearchResult
from src.hallucination_detection.enhanced_searcher import EnhancedIssueSearcher
from src.hallucination_detection.enhanced_reporting import EnhancedReportGenerator
from src.hallucination_detection.threshold_manager import ThresholdManager
from src.detection.keyword_generator import generate_keywords_for_topic

# --- 로깅 설정 ---
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO", colorize=True)
logger.add("logs/bot.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO", encoding="utf-8")
logger.add("logs/error.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="ERROR", encoding="utf-8")

# --- 봇 클래스 및 이벤트 핸들러 ---
intents = discord.Intents.default()
intents.message_content = True

class IssueMonitorBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        logger.info("🤖 IssueMonitorBot 인스턴스 생성됨")

    async def setup_hook(self):
        """봇이 Discord에 로그인한 후, 실행 준비를 위해 호출되는 비동기 메서드."""
        logger.info("⚙️ 봇 셋업 시작: 슬래시 명령어 동기화 시도...")
        try:
            # 글로벌 명령어 동기화 (모든 서버)
            synced = await self.tree.sync()
            logger.success(f"✅ 글로벌 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")

            # 동기화된 명령어 목록 출력
            for command in synced:
                logger.info(f"  - /{command.name}: {command.description}")

        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")
            logger.error(f"상세 오류: {type(e).__name__}: {str(e)}")

    async def on_ready(self):
        """봇이 성공적으로 Discord에 연결되고 모든 준비를 마쳤을 때 호출됩니다."""
        logger.success(f"🎉 {self.user}가 Discord에 성공적으로 연결되었습니다!")
        logger.info(f"📊 봇이 {len(self.guilds)}개 서버에 참여 중입니다.")

        # 참여 중인 서버 목록 출력
        for guild in self.guilds:
            logger.info(f"  - {guild.name} (ID: {guild.id}, 멤버: {guild.member_count}명)")

        # 봇의 '활동' 메시지를 설정하여 현재 상태를 표시
        status_message = f"/monitor (Stage {config.get_current_stage()} 활성화)"
        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name=status_message)
        )
        logger.info(f"👀 봇 상태 설정: '{status_message}'")

    async def on_error(self, event, *args, **kwargs):
        """처리되지 않은 이벤트 관련 오류가 발생했을 때 로깅합니다."""
        logger.error(f"❌ 처리되지 않은 이벤트 오류 발생 ({event}): {args} {kwargs}")

    async def on_application_command_error(self, interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
        """슬래시 명령어 실행 중 오류 발생 시 처리합니다."""
        logger.error(f"❌ 슬래시 명령어 오류 발생: {error}")

        if not interaction.response.is_done():
            await interaction.response.send_message(
                f"❌ 명령어 실행 중 오류가 발생했습니다: {str(error)}",
                ephemeral=True
            )

# 전역 봇 인스턴스 생성
bot = IssueMonitorBot()

# --- 헬퍼 함수 ---
def parse_time_period(period_str: str) -> tuple[datetime, str]:
    """'1주일', '3일' 등 자연어 시간 문자열을 파싱합니다."""
    period_str = period_str.strip().lower()
    now = datetime.now()
    # 정규식을 사용하여 숫자와 단위를 분리
    match = re.match(r'(\d+)\s*(일|주일|개월|달|시간)', period_str)

    if not match:
        # 유효한 형식이 아니면 기본값(최근 1주일) 반환
        return now - timedelta(weeks=1), "최근 1주일"

    number = int(match.group(1))
    unit = match.group(2)

    # 단위에 따라 적절한 시간 차이를 계산
    if unit == '일':
        return now - timedelta(days=number), f"최근 {number}일"
    if unit == '주일':
        return now - timedelta(weeks=number), f"최근 {number}주일"
    if unit in ['개월', '달']:
        return now - timedelta(days=number * 30), f"최근 {number}개월"
    if unit == '시간':
        return now - timedelta(hours=number), f"최근 {number}시간"

    # 예외 처리: 기본값 반환
    return now - timedelta(weeks=1), "최근 1주일"

def validate_topic(topic: str) -> bool:
    """주제 입력값이 유효한지(2글자 이상) 검사합니다."""
    return topic is not None and len(topic.strip()) >= 2

# --- 슬래시 명령어 (수정된 버전) ---
@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링하고 환각 현상을 검증합니다.")
async def monitor_command(interaction: discord.Interaction, 주제: str, 기간: str = "1주일"):
    """이슈 모니터링 메인 명령어 (PDF 보고서 생성 포함).

    사용자로부터 주제와 기간을 입력받아 키워드 생성, 이슈 검색, 환각 탐지,
    보고서 생성의 전체 파이프라인을 실행하고 결과를 Discord에 전송합니다.
    마크다운과 PDF 두 가지 형식의 보고서를 생성합니다.

    Args:
        interaction (discord.Interaction): 사용자의 상호작용 객체.
        topic (str): 분석할 주제어 (예: '양자 컴퓨팅').
        period (str): 검색할 기간 (예: '3일', '2주일'). 기본값은 '1주일'.
    """
    # 기존 변수명 호환성을 위한 변수 할당
    topic = 주제
    period = 기간

    user = interaction.user
    logger.info(f"📝 /monitor 명령어 수신: 사용자='{user.name}', 주제='{topic}', 기간='{period}'")
    await interaction.response.defer(thinking=True)

    try:
        # 주제 유효성 검사
        if not validate_topic(주제):
            await interaction.followup.send("❌ 주제를 2글자 이상 입력해주세요.", ephemeral=True)
            return

        # 기간 파싱
        _, period_description = parse_time_period(기간)

        # 초기 진행 상황 메시지 전송
        progress_embed = discord.Embed(
            title="🔍 이슈 모니터링 시작 (3단계 환각 탐지 활성화)",
            description=f"**주제**: {topic}\n**기간**: {period_description}\n\n⏳ 처리 중...",
            color=0x00aaff,
            timestamp=datetime.now()
        )
        await interaction.followup.send(embed=progress_embed)

        # 진행 상황 업데이트 함수
        async def update_progress(stage: int, message: str):
            progress_embed.description = (
                f"**주제**: {topic}\n**기간**: {period_description}\n\n"
                f"{stage}/5. {message}"
            )
            await interaction.edit_original_response(embed=progress_embed)

        # Performance: Start processing with streaming updates
        start_time = time.time()
        
        # 1. 키워드 생성 (with streaming progress)
        await update_progress(1, "AI 키워드 생성 중...")
        keyword_task = asyncio.create_task(generate_keywords_for_topic(주제))
        
        # Show streaming progress for keyword generation
        while not keyword_task.done():
            elapsed = time.time() - start_time
            await update_progress(1, f"AI 키워드 생성 중... ({elapsed:.1f}초 경과)")
            await asyncio.sleep(2)  # Update every 2 seconds
        
        keyword_result = await keyword_task
        
        # Show intermediate results to user
        preview_keywords = ", ".join(keyword_result.primary_keywords[:3])
        await update_progress(1, f"✅ 키워드 생성 완료: {preview_keywords} 등 {len(keyword_result.primary_keywords)}개")
        await asyncio.sleep(1)  # Brief pause to show completion

        # 2. 환각 탐지가 통합된 검색기 실행 (with streaming progress)
        await update_progress(2, "이슈 검색 및 환각 탐지 중...")
        enhanced_searcher = EnhancedIssueSearcher()
        search_task = asyncio.create_task(enhanced_searcher.search_with_validation(keyword_result, period_description))
        
        # Show streaming progress for search
        search_start = time.time()
        while not search_task.done():
            elapsed = time.time() - search_start
            await update_progress(2, f"이슈 검색 및 환각 탐지 중... ({elapsed:.1f}초 경과)")
            await asyncio.sleep(3)  # Update every 3 seconds for longer operations
        
        search_result = await search_task
        
        # Show search results preview
        issue_count = len(search_result.issues)
        await update_progress(2, f"✅ {issue_count}개 이슈 발견 및 검증 완료")
        await asyncio.sleep(1)

        # 3. 향상된 보고서 생성 (마크다운 + PDF)
        await update_progress(3, "마크다운 보고서 생성 중...")
        from src.hallucination_detection.enhanced_reporting_with_pdf import generate_all_reports

        # PDF 생성 가능 여부 확인
        can_generate_pdf = config.get_openai_api_key() is not None
        if not can_generate_pdf:
            logger.warning("OpenAI API 키가 없어 PDF 생성을 건너뜁니다.")
            await update_progress(3, "보고서 생성 중... (PDF 생성 불가 - OpenAI API 키 필요)")
        else:
            await update_progress(3, "보고서 생성 중... (마크다운 + PDF)")

        # 보고서 생성
        result_embed, markdown_path, pdf_path = await generate_all_reports(
            search_result,
            주제,
            generate_pdf=can_generate_pdf
        )

        # 4. 파일 준비
        await update_progress(4, "파일 첨부 준비 중...")
        files_to_send = []

        # 마크다운 파일 추가
        with open(markdown_path, 'rb') as f:
            markdown_file = discord.File(
                f,
                filename=f"{topic}_보고서_{datetime.now().strftime('%Y%m%d')}.md"
            )
            files_to_send.append(markdown_file)

        # PDF 파일 추가 (있는 경우)
        if pdf_path:
            with open(pdf_path, 'rb') as f:
                pdf_file = discord.File(
                    f,
                    filename=f"{topic}_보고서_{datetime.now().strftime('%Y%m%d')}.pdf"
                )
                files_to_send.append(pdf_file)
            logger.info("PDF 보고서가 성공적으로 생성되었습니다.")

        # 5. 최종 결과 전송
        await update_progress(5, "결과 전송 중...")

        # 파일 형식에 따른 안내 메시지 추가
        if pdf_path:
            file_info = "📎 **첨부 파일**: 마크다운(.md) 및 PDF 보고서"
        else:
            file_info = "📎 **첨부 파일**: 마크다운(.md) 보고서\n" \
                        "💡 *PDF 생성을 위해서는 OpenAI API 키 설정이 필요합니다.*"

        # 결과 임베드에 파일 정보 추가
        if not any(field.name == "📎 첨부 파일" for field in result_embed.fields):
            result_embed.add_field(
                name="📎 첨부 파일",
                value=file_info,
                inline=False
            )

        # 최종 메시지 전송
        await interaction.edit_original_response(
            embed=result_embed,
            attachments=files_to_send
        )

        # 성공 로그
        logger.success(
            f"✅ 모니터링 완료 - 주제: {topic}, "
            f"이슈: {search_result.total_found}개, "
            f"파일: {len(files_to_send)}개"
        )

        # 신뢰도 분포 로그
        if hasattr(search_result, 'confidence_distribution'):
            dist = search_result.confidence_distribution
            logger.info(
                f"신뢰도 분포 - "
                f"높음: {dist.get('high', 0)}개, "
                f"보통: {dist.get('moderate', 0)}개, "
                f"낮음: {dist.get('low', 0)}개"
            )

    except Exception as e:
        logger.error(f"💥 /monitor 명령어 처리 중 심각한 오류 발생: {e}", exc_info=True)

        # 오류 임베드 생성
        error_embed = discord.Embed(
            title="❌ 시스템 오류 발생",
            description=f"요청 처리 중 문제가 발생했습니다.\n\n"
                        f"**오류 내용**: `{str(e)}`\n\n"
                        f"문제가 지속되면 관리자에게 문의해주세요.",
            color=0xff0000,
            timestamp=datetime.now()
        )

        # 오류 타입에 따른 추가 안내
        if "openai" in str(e).lower():
            error_embed.add_field(
                name="💡 해결 방법",
                value="OpenAI API 키 설정을 확인해주세요.",
                inline=False
            )
        elif "perplexity" in str(e).lower():
            error_embed.add_field(
                name="💡 해결 방법",
                value="Perplexity API 키 설정을 확인해주세요.",
                inline=False
            )

        # defer 상태에 따른 응답 방식 선택
        if interaction.is_done():
            await interaction.followup.send(embed=error_embed, ephemeral=True)
        else:
            await interaction.edit_original_response(embed=error_embed)

# --- 추가된 디버깅 및 유틸리티 명령어들 ---

@bot.tree.command(name="debug", description="봇 상태 및 등록된 명령어를 확인합니다.")
async def debug_command(interaction: discord.Interaction):
    """봇의 상태 및 등록된 명령어 목록을 보여줍니다."""
    commands = [cmd.name for cmd in bot.tree.get_commands()]

    embed = discord.Embed(
        title="🔧 디버그 정보",
        color=0x00ff00,
        timestamp=datetime.now()
    )

    embed.add_field(
        name="📋 등록된 명령어",
        value=f"```{', '.join(commands) if commands else '없음'}```",
        inline=False
    )

    embed.add_field(
        name="🌐 네트워크 상태",
        value=f"지연시간: {round(bot.latency * 1000)}ms",
        inline=True
    )

    embed.add_field(
        name="📚 라이브러리 버전",
        value=f"Discord.py: {discord.__version__}",
        inline=True
    )

    embed.add_field(
        name="🏠 서버 정보",
        value=f"참여 중인 서버: {len(bot.guilds)}개",
        inline=True
    )

    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="invite", description="봇 초대 링크를 생성합니다.")
async def invite_command(interaction: discord.Interaction):
    """봇을 다른 서버에 초대할 수 있는 링크를 생성합니다."""
    permissions = discord.Permissions(
        send_messages=True,
        attach_files=True,
        embed_links=True,
        use_slash_commands=True,
        read_message_history=True
    )

    invite_url = discord.utils.oauth_url(
        bot.user.id,
        permissions=permissions,
        scopes=['bot', 'applications.commands']
    )

    embed = discord.Embed(
        title="🔗 봇 초대 링크",
        description=f"[여기를 클릭하여 봇을 서버에 초대하세요]({invite_url})",
        color=0x00aaff
    )

    embed.add_field(
        name="⚠️ 주의사항",
        value="봇이 정상 작동하려면 다음 권한이 필요합니다:\n"
              "• 메시지 보내기\n"
              "• 파일 첨부\n"
              "• 링크 임베드\n"
              "• 슬래시 명령어 사용\n"
              "• 메시지 기록 보기",
        inline=False
    )

    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="봇 시스템의 현재 설정 상태를 확인합니다.")
async def status_command(interaction: discord.Interaction):
    """봇의 API 키 설정 상태 및 활성화된 기능 단계를 보여줍니다."""
    stage = config.get_current_stage()
    embed = discord.Embed(
        title="📊 시스템 상태",
        description=f"현재 실행 가능한 최고 단계는 **{stage}단계**입니다.",
        color=0x00ff00
    )
    stage_info = config.get_stage_info()

    # API 키 설정 상태
    embed.add_field(name="1단계: Discord Bot", value="✅" if stage_info['stage1_discord'] else "❌", inline=True)
    embed.add_field(name="2단계: 키워드 생성 (OpenAI)", value="✅" if stage_info['stage2_openai'] else "❌", inline=True)
    embed.add_field(name="3/4단계: 이슈 검색 (Perplexity)", value="✅" if stage_info['stage3_perplexity'] else "❌",
                    inline=True)

    # 환각 탐지 시스템 상태
    if stage >= 4:
        embed.add_field(
            name="🛡️ 환각 탐지 시스템",
            value=(
                "✅ **3단계 교차 검증 활성화**\n"
                "• RePPL 탐지기\n"
                "• 자기 일관성 검사기\n"
                "• LLM-as-Judge"
            ),
            inline=False
        )

    # PDF 생성 기능 상태 추가
    pdf_status = "✅ 활성화" if config.get_openai_api_key() else "❌ 비활성화 (OpenAI API 키 필요)"
    embed.add_field(
        name="📄 PDF 보고서 생성",
        value=pdf_status,
        inline=False
    )

    # 추가 기능 안내
    if not config.get_openai_api_key():
        embed.add_field(
            name="💡 팁",
            value="OpenAI API 키를 설정하면 LLM으로 개선된 PDF 보고서를 생성할 수 있습니다.",
            inline=False
        )

    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="help", description="봇 사용법을 안내합니다.")
async def help_command(interaction: discord.Interaction):
    """봇의 사용법과 명령어를 안내합니다."""
    embed = discord.Embed(
        title="🤖 이슈 모니터링 봇 사용법",
        color=0x0099ff,
        description="최신 기술 이슈를 모니터링하고 신뢰도 높은 정보를 제공합니다."
    )

    embed.add_field(
        name="`/monitor`",
        value=(
            "`topic`(주제)와 `period`(기간)을 입력하여 이슈를 검색하고 분석합니다.\n"
            "• `topic`: '양자 컴퓨팅', 'AI 윤리' 등\n"
            "• `period`: '3일', '1주일', '2개월' 등 (기본값: '1주일')"
        ),
        inline=False
    )

    embed.add_field(
        name="`/status`",
        value="봇의 현재 설정 상태와 실행 가능한 단계를 확인합니다.",
        inline=False
    )

    embed.add_field(
        name="`/debug`",
        value="봇의 상태와 등록된 명령어를 확인합니다.",
        inline=False
    )

    embed.add_field(
        name="`/invite`",
        value="봇을 다른 서버에 초대할 수 있는 링크를 생성합니다.",
        inline=False
    )

    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="thresholds", description="현재 환각 탐지 임계값 설정을 확인합니다.")
async def thresholds_command(interaction: discord.Interaction):
    """환각 탐지 시스템의 임계값 설정을 보여줍니다."""
    tm = ThresholdManager()
    t = tm.thresholds
    embed = discord.Embed(title="⚙️ 환각 탐지 임계값 설정", color=0x00aaff)
    embed.add_field(name="🎯 시스템 임계값", value=f"최소 신뢰도: {t.min_confidence_threshold:.1%}", inline=False)
    embed.add_field(name="🔍 탐지기별 최소 신뢰도", value=f"• RePPL: {t.reppl_threshold:.1%}\n• 자기 일관성: {t.consistency_threshold:.1%}\n• LLM Judge: {t.llm_judge_threshold:.1%}", inline=True)
    embed.add_field(name="📊 신뢰도 등급", value=f"• 매우 높음: {t.very_high_boundary:.1%} 이상\n• 높음: {t.high_boundary:.1%} 이상\n• 보통: {t.moderate_boundary:.1%} 이상", inline=True)
    await interaction.response.send_message(embed=embed)

# --- 개발용 길드 동기화 함수 (선택사항) ---
async def sync_commands_to_guild(guild_id: int):
    """개발 중 빠른 테스트를 위해 특정 길드에만 명령어를 동기화합니다."""
    try:
        guild = discord.Object(id=guild_id)
        bot.tree.copy_global_to(guild=guild)
        synced = await bot.tree.sync(guild=guild)
        logger.success(f"✅ 길드 {guild_id}에 {len(synced)}개 명령어 동기화 완료")
        return len(synced)
    except Exception as e:
        logger.error(f"❌ 길드별 동기화 실패: {e}")
        return 0

def run_bot():
    """봇을 실행합니다."""
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