import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    # Discord 설정
    DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

    # LLM API 설정
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Perplexity API 설정
    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

    # 데이터베이스 설정
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./discord_bot.db')

    # Redis 설정
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

    # 기타 설정
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # 환각 탐지 설정
    MAX_RETRY_COUNT = 3
    CONFIDENCE_THRESHOLD = 0.7