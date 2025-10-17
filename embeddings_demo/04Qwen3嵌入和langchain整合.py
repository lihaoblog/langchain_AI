from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


#重写HuggingFaceBgeEmbeddings的类
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


if __name__=='__main__':
    qwen3=CustomerQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
    resp=qwen3.embed_documents(
        ['I love you',
         '天气不错'
        ]
    )
    print(resp[0])
    print(len(resp[0]))