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
@bot.tree.command(name="monitor", description="특정 주제에 대한 이슈를 모니터링하고 환각 현상을 검증합니다.")
async def monitor_command(interaction: discord.Interaction, 주제: str, 기간: str = "1주일"):
    logger.info(f"📝 /monitor 수신: 사용자='{interaction.user.name}', 주제='{주제}'")
    await interaction.response.defer(thinking=True)
    try:
        if not validate_topic(주제):
            await interaction.followup.send("❌ 주제를 2글자 이상 입력해주세요.", ephemeral=True)
            return

        _, period_description = parse_time_period(기간)

        await interaction.followup.send(embed=discord.Embed(title="🔍 이슈 모니터링 시작...", description=f"**주제**: {주제}\n**기간**: {period_description}\n\n1/3. 키워드 생성 중...", color=0x00aaff), wait=True)

        keyword_result = await generate_keywords_for_topic(주제)

        await interaction.edit_original_response(embed=discord.Embed(title="🔍 이슈 모니터링 진행 중...", description=f"**주제**: {주제}\n**기간**: {period_description}\n\n2/3. 이슈 검색 및 환각 탐지 중...", color=0x00aaff))

        enhanced_searcher = EnhancedIssueSearcher()
        search_result = await enhanced_searcher.search_with_validation(keyword_result, period_description)

        await interaction.edit_original_response(embed=discord.Embed(title="🔍 이슈 모니터링 진행 중...", description=f"**주제**: {주제}\n**기간**: {period_description}\n\n3/3. 최종 보고서 생성 중...", color=0x00aaff))

        report_generator = EnhancedReportGenerator()
        result_embed = report_generator.generate_discord_embed(search_result)
        detailed_report = report_generator.generate_detailed_report(search_result)
        file_path = report_generator.save_report_to_file(detailed_report, 주제)

        with open(file_path, 'rb') as f:
            discord_file = discord.File(f, filename=os.path.basename(file_path))
            # 최종 결과는 edit_original_response로 전송해야 함
            await interaction.edit_original_response(embed=result_embed, attachments=[discord_file])

    except Exception as e:
        logger.error(f"💥 /monitor 명령어 처리 중 심각한 오류 발생: {e}", exc_info=True)
        error_embed = discord.Embed(title="❌ 시스템 오류 발생", description=f"요청 처리 중 문제가 발생했습니다.\n`오류: {e}`", color=0xff0000)
        await interaction.followup.send(embed=error_embed, ephemeral=True)

@bot.tree.command(name="status", description="봇 시스템의 현재 설정 상태를 확인합니다.")
async def status_command(interaction: discord.Interaction):
    stage = config.get_current_stage()
    stage_info = config.get_stage_info()
    embed = discord.Embed(title="📊 시스템 상태", description=f"현재 실행 가능한 최고 단계는 **{stage}단계**입니다.", color=0x00ff00)
    embed.add_field(name="1단계: Discord Bot", value="✅" if stage_info['stage1_discord'] else "❌", inline=True)
    embed.add_field(name="2단계: 키워드 생성 (OpenAI)", value="✅" if stage_info['stage2_openai'] else "❌", inline=True)
    embed.add_field(name="3/4단계: 이슈 검색 (Perplexity)", value="✅" if stage_info['stage3_perplexity'] else "❌", inline=True)

    if stage >= 4:
        embed.add_field(name="🛡️ 환각 탐지 시스템", value="✅ **3단계 교차 검증 활성화**\n• RePPL\n• 자기 일관성\n• LLM-as-Judge", inline=False)

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
