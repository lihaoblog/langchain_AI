# 实现从数据库存取历史聊天数据
# 连接数据库的包叫做sqlalchemy,系统上下文128k长度
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_demo.demo1 import llm

# from_messages是列表里面放多个元祖 ,以下三部分组成
# 1.创建提示词基础模板
prompt_template=ChatPromptTemplate.from_messages([
    # 1.系统角色
    ('system','你是一个乐于助人的助手,尽可能回答用户名字，提供的历史聊天记录包含你与客户的聊天信息'),
    # 2.历史数据
    (MessagesPlaceholder(variable_name='chat_history',optional=True)),
    # 这个写法也可以
    # ("placeholder","{chat_history}")
    # 3.用户角色及其变量
    ('human','{input}')
])
# 传给模型
chain=prompt_template | llm
# 存储聊天记录:（内存或关系型数据库或redis数据库）
# 消息分5种：1.SystemMessage（系统返的）,HumanMessage（用户输入的）,AIMessage（模型返回），TollMessage(工具返回的)，InMemoryChatMessageHistory(内存返回的)

# 2.存储聊天记录,可以直接连接数据库也可以连到内存，这里是数据库，重点是这个连接的url去百度一下
def get_session_his(session_id:str):
    """从数据库中的历史消息中返回当前id所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connect_string="oracle+cx_oracle://scott:tiger@127.0.0.1:1521/orcl"
    )



# 3.创建带历史记录功能的处理链
chain_with_chat_message_history=RunnableWithMessageHistory(
    chain,
    get_session_his,
    input_messages_key='input',
    history_messages_key='chat_history'  #这个值要和基础模板的值一样，届时需要检索这个字段
)

res=chain_with_chat_message_history.invoke({"input":"你好，我是李浩"},config={"configurable":{"session_id","ques1"}})
res=chain_with_chat_message_history.invoke({"input":"我是谁"},config={"configurable":{"session_id","ques1"}})