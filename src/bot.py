"""
Discord 봇의 메인 진입점.

Discord API와의 상호작용, 슬래시 명령어 처리, 그리고 다른 비즈니스 로직 모듈들
(키워드 생성, 이슈 검색, 환각 탐지, 보고서 생성)의 전체 흐름을 조율(Orchestration)합니다.
"""

import discord
from discord.ext import commands
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger

# --- 모듈 임포트 ---
from src.config import config
from src.models import KeywordResult, SearchResult
# AttributeError 해결을 위해 실제 import 경로에 맞게 수정
from src.hallucination_detection.enhanced_searcher import EnhancedIssueSearcher
from src.hallucination_detection.enhanced_reporting import EnhancedReportGenerator
from src.hallucination_detection.threshold_manager import ThresholdManager
from src.keyword_generator import generate_keywords_for_topic

# --- 로깅 설정 (이전과 동일) ---
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
        logger.info("⚙️ 봇 셋업 시작: 슬래시 명령어 동기화 시도...")
        try:
            synced = await self.tree.sync()
            logger.success(f"✅ 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")
        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")

    async def on_ready(self):
        logger.success(f"🎉 {self.user}가 Discord에 성공적으로 연결되었습니다!")
        status_message = f"/monitor (Stage {config.get_current_stage()} 활성화)"
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=status_message))
        logger.info(f"👀 봇 상태 설정: '{status_message}'")

bot = IssueMonitorBot()

# --- 헬퍼 함수 (이전과 동일) ---
def parse_time_period(period_str: str) -> tuple[datetime, str]:
    period_str = period_str.strip().lower()
    now = datetime.now()
    match = re.match(r'(\d+)\s*(일|주일|개월|달|시간)', period_str)
    if not match: return now - timedelta(weeks=1), "최근 1주일"
    number, unit = int(match.group(1)), match.group(2)
    if unit == '일': return now - timedelta(days=number), f"최근 {number}일"
    if unit == '주일': return now - timedelta(weeks=number), f"최근 {number}주일"
    if unit in ['개월', '달']: return now - timedelta(days=number * 30), f"최근 {number}개월"
    if unit == '시간': return now - timedelta(hours=number), f"최근 {number}시간"
    return now - timedelta(weeks=1), "최근 1주일"

def validate_topic(topic: str) -> bool:
    return topic is not None and len(topic.strip()) >= 2

# --- 슬래시 명령어 ---
bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링하고 환각 현상을 검증합니다.")


async def monitor_command(interaction: discord.Interaction, 주제: str, 기간: str = "1주일"):
    """이슈 모니터링 메인 명령어 (PDF 보고서 생성 포함).

    사용자로부터 주제와 기간을 입력받아 키워드 생성, 이슈 검색, 환각 탐지,
    보고서 생성의 전체 파이프라인을 실행하고 결과를 Discord에 전송합니다.
    마크다운과 PDF 두 가지 형식의 보고서를 생성합니다.

    Args:
        interaction (discord.Interaction): 사용자의 상호작용 객체.
        주제 (str): 분석할 주제어 (예: '양자 컴퓨팅').
        기간 (str): 검색할 기간 (예: '3일', '2주일'). 기본값은 '1주일'.
    """
    user = interaction.user
    logger.info(f"📝 /monitor 명령어 수신: 사용자='{user.name}', 주제='{주제}', 기간='{기간}'")
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
            description=f"**주제**: {주제}\n**기간**: {period_description}\n\n⏳ 처리 중...",
            color=0x00aaff,
            timestamp=datetime.now()
        )
        await interaction.followup.send(embed=progress_embed)

        # 진행 상황 업데이트 함수
        async def update_progress(stage: int, message: str):
            progress_embed.description = (
                f"**주제**: {주제}\n**기간**: {period_description}\n\n"
                f"{stage}/5. {message}"
            )
            await interaction.edit_original_response(embed=progress_embed)

        # 1. 키워드 생성
        await update_progress(1, "AI 키워드 생성 중...")
        keyword_result = await generate_keywords_for_topic(주제)

        # 2. 환각 탐지가 통합된 검색기 실행
        await update_progress(2, "이슈 검색 및 환각 탐지 중...")
        enhanced_searcher = EnhancedIssueSearcher()
        search_result = await enhanced_searcher.search_with_validation(keyword_result, period_description)

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
                filename=f"{주제}_보고서_{datetime.now().strftime('%Y%m%d')}.md"
            )
            files_to_send.append(markdown_file)

        # PDF 파일 추가 (있는 경우)
        if pdf_path:
            with open(pdf_path, 'rb') as f:
                pdf_file = discord.File(
                    f,
                    filename=f"{주제}_보고서_{datetime.now().strftime('%Y%m%d')}.pdf"
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
            f"✅ 모니터링 완료 - 주제: {주제}, "
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


# PDF 보고서 생성 가능 여부를 확인하는 상태 명령어 수정
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

# ... (help, thresholds, run_bot 함수는 이전과 동일)
@bot.tree.command(name="help", description="봇 사용법을 안내합니다.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="🤖 이슈 모니터링 봇 사용법", color=0x0099ff, description="최신 기술 이슈를 모니터링하고 신뢰도 높은 정보를 제공합니다.")
    embed.add_field(name="`/monitor`", value="`주제`와 `기간`을 입력하여 이슈를 검색하고 분석합니다.\n- `주제`: '양자 컴퓨팅'\n- `기간`: '3일' (기본값: '1주일')", inline=False)
    embed.add_field(name="`/status`", value="봇의 현재 설정 상태와 실행 가능한 단계를 확인합니다.", inline=False)
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="thresholds", description="현재 환각 탐지 임계값 설정을 확인합니다.")
async def thresholds_command(interaction: discord.Interaction):
    tm = ThresholdManager()
    t = tm.thresholds
    embed = discord.Embed(title="⚙️ 환각 탐지 임계값 설정", color=0x00aaff)
    embed.add_field(name="🎯 시스템 임계값", value=f"최소 신뢰도: {t.min_confidence_threshold:.1%}", inline=False)
    embed.add_field(name="🔍 탐지기별 최소 신뢰도", value=f"• RePPL: {t.reppl_threshold:.1%}\n• 자기 일관성: {t.consistency_threshold:.1%}\n• LLM Judge: {t.llm_judge_threshold:.1%}", inline=True)
    embed.add_field(name="📊 신뢰도 등급", value=f"• 매우 높음: {t.very_high_boundary:.1%} 이상\n• 높음: {t.high_boundary:.1%} 이상\n• 보통: {t.moderate_boundary:.1%} 이상", inline=True)
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
        logger.critical(f"💥 봇 실행에 실패했습니다: {e}", exc_info=True)

if __name__ == "__main__":
    run_bot()
