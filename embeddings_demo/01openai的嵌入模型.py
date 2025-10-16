# 不用openai向量化，用langchain框架

from langchain_openai import OpenAIEmbeddings
from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

openai_embedding=OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model='text-embedding-3-large',
    dimensions=512,
)

# embed_documents是一个输入文本文件的函数，这里面放文本进行向量化处理
resp=openai_embedding.embed_documents(
    ['I love you',
     '天气不错'
    ]
)
print(resp[0])