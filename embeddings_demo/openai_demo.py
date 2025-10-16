from openai import OpenAI
from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

#创建一个openai的连接
client=OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

text="I am LI"
#向量化一个文本
respond=client.embeddings.create(
    model='text-embedding-3-large',
    dimensions=512,
    input=text
)
#输出向量化之后的文本和大小
print(respond.data[0].embedding)
print(len(respond.data[0].embedding))