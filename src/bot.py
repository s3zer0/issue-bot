"""
Discord 봇의 메인 진입점.

Discord API와의 상호작용, 슬래시 명령어 처리, 그리고 다른 비즈니스 로직 모듈들
(키워드 생성, 이슈 검색, 환각 탐지, 보고서 생성)의 전체 흐름을 조율(Orchestration)합니다.
"""

import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta
import re
import sys
import os
from loguru import logger

# --- 모듈 임포트 ---
from src.config import config  # 환경 설정 관리 모듈
from src.models import KeywordResult, SearchResult  # 데이터 모델 클래스
from src.hallucination_detector import RePPLEnhancedIssueSearcher  # 환각 탐지 및 이슈 검색 모듈
from src.keyword_generator import generate_keywords_for_topic  # 키워드 생성 모듈
from src.reporting import (
    format_search_summary,  # 검색 결과 요약 포맷팅 함수
    create_detailed_report_from_search_result,  # 상세 보고서 생성 함수
    save_report_to_file  # 보고서 파일 저장 함수
)
from src.hallucination_detection.enhanced_reporting import EnhancedReportGenerator
from src.hallucination_detection.threshold_manager import ThresholdManager, ConfidenceLevel


# --- 로깅 설정 ---
# 로그 디렉토리 생성 (없을 경우)
os.makedirs("logs", exist_ok=True)
logger.remove()  # 기본 로거 제거

# 콘솔 로그 설정: 실시간 디버깅용으로 색상과 함께 출력
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",  # 디버그 레벨 이상 로그 출력
    colorize=True  # 콘솔에 색상 적용
)

# 파일 로그 설정: 상세 기록을 파일에 저장
log_file = "logs/bot.log"
if os.path.exists(log_file):
    try:
        os.remove(log_file)  # 기존 로그 파일 삭제
    except OSError as e:
        logger.error(f"로그 파일 삭제 실패: {e}")  # 삭제 실패 시 에러 로그
logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",  # 정보 레벨 이상 로그 기록
    encoding="utf-8"  # 한글 지원을 위해 UTF-8 인코딩
)

# 에러 로그 설정: 오류만 별도로 기록
error_log_file = "logs/error.log"
if os.path.exists(error_log_file):
    try:
        os.remove(error_log_file)  # 기존 에러 로그 파일 삭제
    except OSError as e:
        logger.error(f"에러 로그 파일 삭제 실패: {e}")  # 삭제 실패 시 에러 로그
logger.add(
    error_log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",  # 에러 레벨 로그만 기록
    encoding="utf-8"
)

# --- 봇 클래스 및 이벤트 핸들러 ---
intents = discord.Intents.default()
intents.message_content = True  # 메시지 콘텐츠 접근 권한 활성화


class IssueMonitorBot(commands.Bot):
    """Discord 봇 클라이언트 클래스.

    discord.ext.commands.Bot을 상속받아, 봇의 생명주기(lifecycle)와 관련된
    이벤트 핸들러(on_ready, setup_hook 등)를 정의합니다.
    """

    def __init__(self):
        """IssueMonitorBot 인스턴스를 초기화합니다."""
        # 명령어 접두사는 '!', 슬래시 명령어 사용을 위해 intents와 help_command 설정
        super().__init__(command_prefix='!', intents=intents, help_command=None)
        logger.info("🤖 IssueMonitorBot 인스턴스 생성됨")

    async def setup_hook(self):
        """봇이 Discord에 로그인한 후, 실행 준비를 위해 호출되는 비동기 메서드.

        슬래시 명령어를 Discord 서버와 동기화하는 역할을 합니다.
        """
        logger.info("⚙️ 봇 셋업 시작: 슬래시 명령어 동기화 시도...")
        try:
            synced = await self.tree.sync()  # 슬래시 명령어 동기화
            logger.success(f"✅ 슬래시 명령어 동기화 완료: {len(synced)}개 명령어")
        except Exception as e:
            logger.error(f"❌ 슬래시 명령어 동기화 실패: {e}")

    async def on_ready(self):
        """봇이 성공적으로 Discord에 연결되고 모든 준비를 마쳤을 때 호출됩니다."""
        logger.success(f"🎉 {self.user}가 Discord에 성공적으로 연결되었습니다!")
        logger.info(f"📊 봇이 {len(self.guilds)}개 서버에 참여 중입니다.")

        # 봇의 '활동' 메시지를 설정하여 현재 상태를 표시
        status_message = f"/monitor (Stage {config.get_current_stage()} 활성화)"
        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name=status_message)
        )
        logger.info(f"👀 봇 상태 설정: '{status_message}'")

    async def on_error(self, event, *args, **kwargs):
        """처리되지 않은 이벤트 관련 오류가 발생했을 때 로깅합니다."""
        logger.error(f"❌ 처리되지 않은 이벤트 오류 발생 ({event}): {args} {kwargs}")


# 전역 봇 인스턴스 생성
bot = IssueMonitorBot()


# --- 헬퍼 함수 ---
def parse_time_period(period_str: str) -> tuple[datetime, str]:
    """'1주일', '3일' 등 자연어 시간 문자열을 파싱합니다.

    Args:
        period_str (str): 파싱할 시간 문자열 (예: '3일', '2주일').

    Returns:
        tuple[datetime, str]: 검색 시작 날짜(datetime 객체)와 파싱된 기간 설명 문자열.
    """
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
    """주제 입력값이 유효한지(2글자 이상) 검사합니다.

    Args:
        topic (str): 검사할 주제 문자열.

    Returns:
        bool: 주제가 유효하면 True, 아니면 False.
    """
    return topic is not None and len(topic.strip()) >= 2


# --- 슬래시 명령어 ---
@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링하고 환각 현상을 검증합니다.")
async def monitor_command(interaction: discord.Interaction, 주제: str, 기간: str = "1주일"):
    """이슈 모니터링 메인 명령어 (향상된 버전).

    사용자로부터 주제와 기간을 입력받아 키워드 생성, 이슈 검색, 환각 탐지,
    보고서 생성의 전체 파이프라인을 실행하고 결과를 Discord에 전송합니다.

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

        # 1. 키워드 생성
        keyword_result = await generate_keywords_for_topic(주제)

        # 2. 환각 탐지가 통합된 검색기 실행
        enhanced_searcher = RePPLEnhancedIssueSearcher()
        search_result = await enhanced_searcher.search_with_validation(keyword_result, period_description)

        # 3. 향상된 보고서 생성
        report_generator = EnhancedReportGenerator()

        # Discord 임베드 생성
        result_embed = report_generator.generate_discord_embed(search_result)

        # 상세 보고서 생성 및 저장
        detailed_report = report_generator.generate_detailed_report(search_result)
        file_path = report_generator.save_report_to_file(detailed_report, 주제)

        # 결과 전송
        with open(file_path, 'rb') as f:
            discord_file = discord.File(f, filename=os.path.basename(file_path))
            await interaction.followup.send(embed=result_embed, file=discord_file)

        # 신뢰도 분포 로그
        if hasattr(search_result, 'confidence_distribution'):
            dist = search_result.confidence_distribution
            logger.info(
                f"✅ 모니터링 완료 - 신뢰도 분포: "
                f"높음({dist['high']}), 보통({dist['moderate']}), 낮음({dist['low']})"
            )

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
    """봇의 기능과 명령어 사용법에 대한 도움말을 제공합니다.

    Args:
        interaction (discord.Interaction): 사용자의 상호작용 객체.
    """
    embed = discord.Embed(
        title="🤖 이슈 모니터링 봇 사용법",
        color=0x0099ff,
        description="이 봇은 최신 기술 이슈를 모니터링하고 LLM의 환각 현상을 최소화하여 신뢰도 높은 정보를 제공합니다."
    )
    embed.add_field(
        name="`/monitor`",
        value="`주제`와 `기간`을 입력하여 이슈를 검색하고 분석합니다.\n- `주제`: '양자 컴퓨팅', 'AI 반도체' 등\n- `기간`: '3일', '2주일', '1개월' 등",
        inline=False
    )
    embed.add_field(name="`/status`", value="봇의 현재 설정 상태와 실행 가능한 단계를 확인합니다.", inline=False)
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

    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="thresholds", description="현재 환각 탐지 임계값 설정을 확인합니다.")
async def thresholds_command(interaction: discord.Interaction):
    """현재 환각 탐지 시스템의 임계값 설정을 보여줍니다."""
    threshold_manager = ThresholdManager()

    embed = discord.Embed(
        title="⚙️ 환각 탐지 임계값 설정",
        description="현재 시스템에 설정된 신뢰도 임계값입니다.",
        color=0x00aaff
    )

    # 전체 시스템 임계값
    embed.add_field(
        name="🎯 시스템 임계값",
        value=f"최소 신뢰도: {threshold_manager.thresholds.min_confidence_threshold:.1%}",
        inline=False
    )

    # 개별 탐지기 임계값
    embed.add_field(
        name="🔍 탐지기별 최소 신뢰도",
        value=(
            f"• RePPL: {threshold_manager.thresholds.reppl_threshold:.1%}\n"
            f"• 자기 일관성: {threshold_manager.thresholds.consistency_threshold:.1%}\n"
            f"• LLM Judge: {threshold_manager.thresholds.llm_judge_threshold:.1%}"
        ),
        inline=True
    )

    # 신뢰도 등급 경계
    embed.add_field(
        name="📊 신뢰도 등급",
        value=(
            f"• 매우 높음: {threshold_manager.thresholds.very_high_boundary:.1%} 이상\n"
            f"• 높음: {threshold_manager.thresholds.high_boundary:.1%} 이상\n"
            f"• 보통: {threshold_manager.thresholds.moderate_boundary:.1%} 이상\n"
            f"• 낮음: {threshold_manager.thresholds.low_boundary:.1%} 이상"
        ),
        inline=True
    )

    # 보고서 옵션
    embed.add_field(
        name="📄 보고서 옵션",
        value=(
            f"• 낮은 신뢰도 포함: {'예' if threshold_manager.thresholds.include_low_confidence else '아니오'}\n"
            f"• 상세 분석 최소 신뢰도: {threshold_manager.thresholds.detailed_analysis_threshold:.1%}"
        ),
        inline=False
    )

    await interaction.response.send_message(embed=embed)


# --- 봇 실행 ---
def run_bot():
    """봇을 실행하는 메인 함수."""
    discord_token = config.get_discord_token()
    if not discord_token:
        # Discord 토큰이 없으면 실행 중단
        logger.critical("❌ Discord 봇 토큰이 없습니다. .env 파일을 확인해주세요!")
        return

    try:
        logger.info("🚀 Discord 봇을 시작합니다...")
        bot.run(discord_token, log_handler=None)  # Discord 봇 실행
    except Exception as e:
        # 실행 실패 시 에러 로그 기록
        logger.critical(f"💥 봇 실행에 실패했습니다: {e}", exc_info=True)


if __name__ == "__main__":
    run_bot()  # 스크립트 직접 실행 시 봇 시작