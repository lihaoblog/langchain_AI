from langchain.embeddings import HuggingFaceBgeEmbeddings
#改成BAAI/bge-small-zh-v1.5小版本更快一些
# 第一次配置会自动下载模型，默认在C盘用户cache下，可以修改环境变量来改下载位置HF_HOME=指定位置
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}  #改成cpu处理
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
bge_hf_embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)


resp=bge_hf_embedding.embed_documents(
    ['I love you',
     '天气不错'
    ]
)
print(resp[0])