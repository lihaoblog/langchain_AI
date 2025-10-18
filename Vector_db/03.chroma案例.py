from langchain_chroma import Chroma
from langchain_core.documents import Document
from Vector_db.qwen_custom_embedding import CustomerQwen3Embeddings



# 写入向量化工具
qwen_embedding=CustomerQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
# 加载向量化数据库,分别给数据库取个名字，向量化工具，存放地址
vector_store=Chroma(
    collection_name='t_news',
    embedding_function=qwen_embedding,
    persist_directory=r'C:\Users\lihao\Desktop\dify\chroma_db'
)

document1=Document(
    #page_content，也只有需要向量化的数据才能加索引
    page_content="早上吃包子",
    metadata={"source":"sweet"},
)

document2=Document(
    page_content="中午吃米饭",
    metadata={"source":"news"},
)

document3=Document(
    page_content="晚上不吃",
    metadata={"source":"olds"},
)

documents=[
    document1,
    document2,
    document3,
]

# 增加一些id
ids=['id'+str(i+1) for i in range(len(documents))]
# 存入数据库，带上id
vector_store.add_documents(documents,ids=ids)

resp=vector_store.similarity_search_with_score('什么时候不吃',k=3,filter={"source":"news"})

for r,score in resp:
    print(r.id)
    print(f"*{score},{r.page_content},{r.metadata}")