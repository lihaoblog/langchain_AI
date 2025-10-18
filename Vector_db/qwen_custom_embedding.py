# 放千问的embedding
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class CustomerQwen3Embeddings(Embeddings):
    """自定义一个langchain和embedding的类"""

    def __init__(self,model_name):
        self.qwen3_embedding = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    # 用户问题向量化
    def embed_query(self,text:str)->list[float]:
        return self.embed_documents([text])[0]

    # 对企业数据向量化
    def embed_documents(self,texts:list[str])->list[list[str]]:
        return self.qwen3_embedding.encode(texts)