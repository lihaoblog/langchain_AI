import pandas as pd
import numpy as np
from langchain.embeddings import HuggingFaceBgeEmbeddings
import ast

#改成BAAI/bge-small-zh-v1.5小版本更快一些
# 第一次配置会自动下载模型，默认在C盘用户cache下，可以修改环境变量来改下载位置HF_HOME=指定位置
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}  #改成cpu处理
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
bge_hf_embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 写一个函数来使用该模型
def text_2_embedding(text):
    resp=bge_hf_embedding.embed_documents(
        [text]
    )
    return resp[0]

#1.读取数据
def embedding_2_file(source_file,output_file):
    """读取csv文件，用embedding模型向量化到新文件"""
    #df = pd.read_csv(source_file,index_col=0)  #指定从哪一列开始
    df = pd.read_csv(source_file)
    df=df[['time','product','use_id','score','summary','text']]

    print(df.head(2))
#2.清洗数据并合并
    df=df.dropna()
    df['text_content']='summary:'+df.summary.str.strip()+";text:"+df.text.str.strip()
    print(df.head(2))
#3.向量化存到新文件里，这个apply是表示对前面的每一个都执行括号里的
    df['embedding']=df.text_content.apply(lambda x : text_2_embedding(x))
    df.to_csv(output_file)

# 求向量余弦夹角，用到numpy,就是向量积除以向量模的积
def cosine_distance(a,b):
    """计算余弦距离"""
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def search_text(input,embedding_file,top_n=3):
    """根据用户提出问题进行语义检索"""
    df_date = pd.read_csv(embedding_file)
    # 把字符串变成向量，保持到新字段
    df_date['embedding_vector']=df_date['embedding'].apply(ast.literal_eval)
    #处理新输入的数据
    input_vector=text_2_embedding(input)
    df_date['similarity']=df_date.embedding_vector.apply(lambda x:cosine_distance(x,input_vector))

    # 这个括号阔起来，里面的df_date就可以不用写了，直接加.就好了
    res=(
        df_date.sort_values('similarity',ascending=False)
        .head(top_n)
        .text_content.str.replace('summary:','')
        .replace(';text:','')
    )

    for i in res:
        print(i)
        print("*"*50)

if __name__=='__main__':
    # embedding_2_file(r'C:\Users\lihao\Desktop\dify\review_data_sample.csv',r'C:\Users\lihao\Desktop\dify\output.csv')
    search_text('I like you',r'C:\Users\lihao\Desktop\dify\output.csv')
    print('success')
