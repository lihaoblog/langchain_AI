#聊天样式的提示词模板
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder, \
    PromptTemplate

from langchain_demo.demo1 import llm

# 1给例子
examples=[
    {"input":"2@2","output":"4"},
    {"input":"2@3","output":"8"},
    ]

# 2定输出格式，单个输入出的模板
base_prompt=ChatPromptTemplate.from_messages(
    [
        ('human','{input}'),
        ('ai','{output}'),
    ]
)


#还可以这样写提示词模板
prompt1=(
    #甚至PromptTemplate.from_template都可以不要
    PromptTemplate.from_template("给一个主题{topic}写报幕词")
    +",要求1.内容搞笑"
    +"，2.采用{language}输出"
)


# 包含示例的模板
few_shot_prompt=FewShotChatMessagePromptTemplate(
    examples=examples,  #给回答原理
    example_prompt=base_prompt,  #给回答模板
)

# 给上一些其他补充信息
final_template= ChatPromptTemplate(
    [
        ("system","你是一个智能ai助手"),
        few_shot_prompt,
        MessagesPlaceholder("msgs")
    ]
)


# chain=final_template | llm

# chain=prompt1 | llm

#加上输出解释器
chain=prompt1 | llm | StrOutputParser()
# res=chain.invoke({"msgs":[HumanMessage(content="你好，主持人")]})
# res=chain.invoke({"msgs":[HumanMessage(content="2@6结果是多少")]})
res1=chain.invoke({"topic":"演讲","language":"english"})


# print(res)
print(res1)