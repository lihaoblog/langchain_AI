
#导入的ChatOpenAI所以要用ChatOpenAI实例，除此之外还有langchain自己也有叫ChatAnthropic要install langchain-anthorpic
#其实所有的模型都可以用ChatOpenAI实例来达到同样的标准
from langchain_openai import ChatOpenAI
from env_utils import OPENAI_BASE_URL ,OPENAI_API_KEY ,DEEPSEEK_BASE_URL ,DEEPSEEK_API_KEY,LOCAL_BASE_URL



# 这个是调用gpt的
# llm=ChatOpenAI(
#     model='gpt-4o-mini',
#     temperature = 0.7,
#     base_url=OPENAI_BASE_URL,
#     api_key=OPENAI_API_KEY,
#
# )

# 这个是本地私有化部署的deepseek,需要开代理
llm=ChatOpenAI(
    model='ds-qwen3-8b',
    temperature = 0.7,
    api_key='xx',
    base_url=LOCAL_BASE_URL,
    extra_body={'chat_template_kwargs':{'enable_thing':True}},
)


# 这个是claude,改一下模型名字就好了
# llm=ChatOpenAI(
#     model='claude-sonnet-4-20250514',
#     temperature = 0.7,
#     base_url=OPENAI_BASE_URL,
#     api_key=OPENAI_API_KEY,
#
# )


# 用国内的deepseek
# llm=ChatOpenAI(
#     model='deepseek-reasoner',
#     temperature = 0.7,
#     base_url=DEEPSEEK_BASE_URL,
#     api_key=DEEPSEEK_API_KEY,
#
# )

# 下命令,可以用字典或者二元祖
# message=[('system','你是一个智能助手'),
#          ('human','介绍一下什么是深度学习')
#          ]
#
# #传入命令
# answer=llm.invoke(message)
#
# print(answer)