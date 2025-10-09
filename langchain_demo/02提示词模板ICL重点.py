# in-context Learning（ICL）给一个示例上下文
# 大模型生成的模板，自己来指定一个模板示例   FewShotPromptTemplate少样本提示词
# 模板分三段examples base_template  final_template
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from langchain_demo.demo1 import llm

examples=[{
    "question":"美国总统林肯的兄弟有多少个",
    "answer":"""
    是否需要后续问题：是
    后续问题：林肯的父亲是谁？
    中间答案：林肯父亲是。。
    后续问题：林肯父亲有多少子女？
    后续答案：有XX个子女
    所以最终答案：林肯有n个兄弟"""
}]

# 基本提示词模板：
base_template=PromptTemplate.from_template("问题：{question}\n\n{answer}")

# langchain有自己的链接基础提示词模板和我们给的回答模板连接起来FewShotPromptTemplate实例
final_template=FewShotPromptTemplate(
    examples=examples,                  #传入示例模板
    example_prompt=base_template,       #传入单个示例模板
    suffix="问题：：{input}",             #追加问题的回答模板
    input_variables=["input"],          #指定输入变量
)

#管道符把万平米给的模板给大模型处理
chain= final_template | llm
resp=chain.invoke({"input":"美国总统特朗普有几个兄弟姐妹"})
print(resp)
