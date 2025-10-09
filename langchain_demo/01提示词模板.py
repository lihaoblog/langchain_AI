# 当大模型回答不佳，第一个想到的就是优化提示词，问的准确点
# langchain的表达式语言LCEL
from langchain_core.prompts import PromptTemplate

prompt_template=PromptTemplate.from_template('帮我生成一个简单的，关于{topic}主题的')