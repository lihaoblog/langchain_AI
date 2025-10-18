# 数据写入磁盘和本地,以及本地读出

from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings

from Vector_db.qwen_custom_embedding import CustomerQwen3Embeddings

qwen3=CustomerQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
#向量数据库不等于传统数据库
# 1.初始化数据库和创建索引,这个写法可以获得向量长度，设置动态索引长度
index=faiss.IndexFlatL2(len(qwen3.embed_query('hahahah')))

vector_store=FAISS(
    embedding_function=qwen3,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
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
# 数据写入本地磁盘
vector_store.save_local(r'C:\Users\lihao\Desktop\dify\faiss_db')
# 读取磁盘
vector_store_out=FAISS.load_local(r'C:\Users\lihao\Desktop\dify\faiss_db',embeddings=qwen3,allow_dangerous_deserialization=True)
vector_store.delete(ids=['id3'])
# 数据库的相似查询
result=vector_store_out.similarity_search_with_score("晚上吃什么",k=2)
for r,score in result:
    print(type(r))
    print(f'score={score},{r.page_content},{r.metadata}')