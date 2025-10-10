#聊天提示词模板 ChatPromptTemplate 英文对应
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_demo.demo1 import llm

#聊天模板要加上的
prompt_template=ChatPromptTemplate.from_messages([
    ("system","你是一个幽默的主持人"),
    ("user","生成一个简短的关于{topic}报幕词")
])

# 另一种写法用消息占位符
prompt_template2=ChatPromptTemplate.from_messages([
    ("system","你是一个幽默的主持人"),
    MessagesPlaceholder('msgs')
])
# 给参数写法，用于让用户输入
prompt_template2.invoke({"msgs":[HumanMessage(content="你好，主持人")]})


# 这是直接输出的，没有大模型处理的样子
print(prompt_template.invoke({"topic":"相声"}))
# 传给大模型回答
chain=prompt_template | llm
res= chain.invoke({"topic":"相声"})
print(res)