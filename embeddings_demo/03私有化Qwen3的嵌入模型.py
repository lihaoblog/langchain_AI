from sentence_transformers import SentenceTransformer

#执行会直接安装Qwen/Qwen3-Embedding-0.6B
qwen3_embedding=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
#想做知识库问答、向量检索（配合 LangChain）用 embed_documents()，而encode() —— SentenceTransformer 原生方法
resp=qwen3_embedding.encode(
    [
        "I love you",
        "天气不错"
    ]
)

print(resp)
print(len(resp[0]))
