# 当大模型回答不佳，第一个想到的就是优化提示词，问的准确点
# langchain的表达式语言LCEL，链式表达。通过该方式得到的都是可执行模板
# 所有的可执行模板都可以用invoke关键字
# runlab可执行对象都可以通过invoke来调用。都可放到LCEL模板中
from langchain_core.prompts import PromptTemplate
from demo1 import llm

prompt_template=PromptTemplate.from_template('帮我生成一个简单的，关于{topic}主题的报幕词')

# res=prompt_template.invoke({"topic","相声"})
# print(res)

chain=prompt_template | llm

res=chain.invoke({"topic","相声"})
print(res)