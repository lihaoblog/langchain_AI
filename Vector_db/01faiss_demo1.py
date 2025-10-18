from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings

from Vector_db.qwen_custom_embedding import CustomerQwen3Embeddings

# #改成BAAI/bge-small-zh-v1.5小版本更快一些
# # 第一次配置会自动下载模型，默认在C盘用户cache下，可以修改环境变量来改下载位置HF_HOME=指定位置
# model_name = "BAAI/bge-small-zh-v1.5"
# model_kwargs = {'device': 'cpu'}  #改成cpu处理
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# bge_hf_embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

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

# 数据库的相似查询
result=vector_store.similarity_search("晚上吃什么",k=2)
for r in result:
    print(type(r))
    print(f'{r.page_content},{r.metadata}')