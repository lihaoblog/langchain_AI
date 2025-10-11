# SimpleJsonOutputParser需要一个提示词模板做说明，不需要类（deepseek只支持这种）

from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_demo.demo1 import llm

prompt=ChatPromptTemplate.from_template(
        "尽你所能回答问题"   #基本指令
        
        # 输出格式
        '你必须始终包含一个"answer"和"after_answer"键的json对象,其中"answer"代表对用户的回答，"after_answer"代表问题的衍生问题'
        "{question}"  #问题占位符
)

# runnable=llm.with_sructured_output()

# 传给json解释器输出格式,在python中json输出就是字典格式
# 这个管道符就是LECL
chain=prompt | llm | SimpleJsonOutputParser()


res= chain.invoke({"question":"细胞的动力源是什么?"})
print(res)
