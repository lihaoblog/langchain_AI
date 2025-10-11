from pydantic import BaseModel, Field

from langchain_demo.demo1 import llm


class Respond_Formatter(BaseModel):
    """"用此工具来结构化输出"""
    answer:str=Field(description="对用户问题的回复")
    follow:str=Field(description="用户后续可能提出的问题")

runnable=llm.bind_tools([Respond_Formatter])

res=runnable.invoke("细胞的动力源是什么")
print(res)
res.pretty_print()