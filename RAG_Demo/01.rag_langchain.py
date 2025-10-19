import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Vector_db.qwen_custom_embedding import CustomerQwen3Embeddings
from langchain_demo.demo1 import llm

qwen3=CustomerQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
# 构建向量数据库
vector_store=Chroma(
    collection_name='t_agent_blog',
    embedding_function=qwen3,
    persist_directory=r'C:\Users\lihao\Desktop\dify\rag_db'
)

def create_dense_db():
    """把网络关于agent的博客数据库写入向量数据库"""
    #langchain有一个WebBaseLoader函数来打开网页
    loader=WebBaseLoader(
        web_path=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
        #网址解析器
        bs_kwargs=dict(
            #爬虫常用beautifulsoup解析器，需要指定class
            parse_only=bs4.SoupStrainer(
                #指定要解析的HTML类名
                class_=("post-content","post-title","post-header")
            )
        )
    )

    #使用你定义好的加载器 loader，读取数据源（文件、网页、目录等），并返回一个由 Document 对象组成的 列表。CSVLoader → 读取 CSV 文件
    docs_list=loader.load()
    # 对数据进行切割,块大小1000，重叠200
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # 分割文档
    splits = text_splitter.split_documents(docs_list)

    print('doc的数量为：',len(splits))
    #注意for循环写法
    ids=['id'+str(i+1) for i in range(len(splits))]
    vector_store.add_documents(documents=splits,ids=ids)


# create_dense_db()

### 问题上下文化 ###
# 系统提示词：用于将带有聊天历史的问题转化为独立问题
contextualize_q_system_prompt = (
    "给定聊天历史和最新的用户问题（可能引用聊天历史中的上下文），"
    "将其重新表述为一个独立的问题（不需要聊天历史也能理解）。"
    "不要回答问题，只需在需要时重新表述问题，否则保持原样。"
)

contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [
        'system',contextualize_q_system_prompt,#系统角色提示
        MessagesPlaceholder("chat_history"),#聊天历史占位符
        "human","{input}"                   #用户输入占位符
    ]
)

#创建一个向量数据库的检索器,把之前的检索器来作为这次的
retriever=vector_store.as_retriever(search_kwargs={'k':2})

#创建一个上下文感知的检索器，需要三个值，大模型，向量库检索器，提示词
history_aware_retriever=create_history_aware_retriever(
    llm,retriever,contextualize_q_prompt
)

# 定义回答模板和助手行为
system_prompt = (
    "你是一个问答任务助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果不知道答案，就说你不知道。"
    "回答最多三句话，保持简洁。"
    "\n\n"
    "{context}"  # 从向量数据库中检索出来的doc
)

# 创建回答提示词模板
qa_prompt=ChatPromptTemplate.from_messages(
    [
        'system',contextualize_q_system_prompt,#系统角色提示
        MessagesPlaceholder("chat_history"),#聊天历史占位符
        "human","{input}"                   #用户输入占位符
    ]
)

#创建文档处理链
question_chain = create_stuff_documents_chain(llm, qa_prompt)
# 创建RAG检索链
rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

store = {}  # 用来保存历史消息, key : 会话ID session_id
def get_session_history(session_id: str):
    """从内存中的历史消息列表中 返回当前会话 的所有历史消息"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建带历史记录功能的处理链
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',  # 输出消息的键
)

# 调用会话式RAG链，询问"什么是任务分解？"
resp1 = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},  # 用户输入问题
    config={
        "configurable": {"session_id": "abc123"}  # 使用会话ID "abc123" 保持对话历史
    }
)

print(resp1['answer'])

resp2 = conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},  # 用户输入问题
    config={
        "configurable": {"session_id": "abc123"}  # 使用会话ID "abc123" 保持对话历史
    }
)

print(resp2['answer'])