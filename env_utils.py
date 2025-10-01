# 加载配置文件
from dotenv import load_dotenv
import os

load_dotenv(override=True)

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY=os.getenv('DEEPSEEK_API_KEY')
OPENAI_BASE_URL=os.getenv('OPENAI_BASE_URL')
DEEPSEEK_BASE_URL=os.getenv('DEEPSEEK_BASE_URL')
LOCAL_BASE_URL=os.getenv('LOCAL_BASE_URL')