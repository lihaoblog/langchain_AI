from typing import Optional
import json

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from langchain_demo.demo1 import llm


# rating: Optional[int] 表示类型可以是int也可以是空

#定义一个pydantic的类，叫数据模型类类似POVO
class joke(BaseModel):
    setup:str=Field(description="笑话开头部分")
    punchline:str=Field(description="笑话的包袱/笑点")
    rating: Optional[int]=Field(description="笑话的好笑程度评分，1-10的评价越高越好")

#给提示词
prompt_template=PromptTemplate.from_template("帮我生成一个{topic}的笑话,50字左右")
# 结构化输出
runnable=llm.with_structured_output(joke)

chain=prompt_template | runnable
res=chain.invoke({"topic","爱情"})

print(res)
print(res.__dict__)
js=json.dumps(res.__dict__)
print(js)