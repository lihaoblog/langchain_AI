# 实现音频输入返回功能
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_demo.demo1 import llm
import gradio as gr

# from_messages是列表里面放多个元祖 ,以下三部分组成
# 1.创建提示词基础模板
prompt_template=ChatPromptTemplate.from_messages([
    # 1.系统角色
    ('system','{system_messages}'),
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
store={}    # 用来保存所有历史数据消息,key:会话id session_id
def get_session_his(session_id:str):
    """从内存的历史消息中返回当前id所有历史消息"""
    if session_id not in store:
        # 没有给个初始值,内存的聊天记录
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3.创建带历史记录功能的处理链
chain_with_chat_message_history=RunnableWithMessageHistory(
    chain,
    get_session_his,
    input_messages_key='input',
    history_messages_key='chat_history'  #这个值要和基础模板的值一样，届时需要检索这个字段
)

# 4.剪辑和摘要上下文,历史数据保留最近2条，其余写成摘要
def summerize_messages(current_inout):
    # 看看id在不在,不在抛出异常
    session_id=current_inout['config']['configurable']['session_id']
    if not session_id:
        raise ValueError('必须通过config参数提供的session_id')

    # id在则取出该id的历史数据
    chat_history=get_session_his(session_id)
    store_messages=chat_history.messages   # 我们只对messages信息进行处理，别的不动,拿出来的是一个列表

    if len(store_messages)<=2:
        return False   # 不足两条不处理
    # 满足两条进行处理，挑出最近也就是最后两条
    last_messages=store_messages[-2:]
    message_need_summerize=store_messages[:-2]

    # 取到数据后让大模型来处理成摘要
    summerize_prompt=ChatPromptTemplate.from_messages(
        [('system','请将下列的信息压缩成一条摘要'),
         ('placeholder','{chat_history}'),
         ('human','请生成包含上面信息的摘要，要求保留关键信息')]
    )
    summerize_chain=summerize_prompt | llm
    #生成摘要
    summerize_resp=summerize_chain.invoke({'chat_history':message_need_summerize})

    # #清空该id所有的数据，然后将2条和摘要写入
    # chat_history.clear()
    # chat_history.add_message(summerize_resp)
    # for msg in last_messages:
    #     chat_history.add_message(msg)
    # return True

    # 直接返回结构化结果,不用clear()，返回摘要和最近两行
    return {
        "original_messages":last_messages,
        "summerize":summerize_resp
    }

# 最终的链,需要三个值传过去，1输入值，2历史值，3系统。都是要从返回值拿
final_chain=(RunnablePassthrough.assign(messages_summerized=summerize_messages) | RunnablePassthrough.assign(
    input=lambda x:x['input'],
    chat_history=lambda x:x['messages_summerized']['original_messages'],  #拿到messages_summerized
    system_messages=lambda x:f"你是一个乐于助人的智能助手，尽可能回答问题。摘要：{x['messages_summerized']['summerize'].content}"
                    if x['messages_summerized'].get('summerize') else "无摘要"
) | chain_with_chat_message_history )


#web界面的核心数据，输入框
def add_message(chat_history,user_message):
    if user_message:
        chat_history.append({'role':'user','content':user_message})
    return chat_history,''    #返回空值是为了在发送信息之后，输入框里面消息消失，然后输入的数据算历史
# 输入后执行链
def execute_chain(chat_history):
    input = chat_history[-1]
    result=final_chain.invoke({'input':input['content'],"config":{"configurable":{"session_id":"ques1"}}},
                              config={"configurable": {"session_id": "ques1"}})
    chat_history.append({'role':'assistant','content':result.content})
    return chat_history

# 实现音频处理
def read_audio(chat_history,audio_message):
    print(audio_message)

# 开发web界面
with gr.Blocks(title="多模态聊天机器人",theme=gr.themes.Soft()) as block:   #开发一个空白页
    #聊天历史记录的组件
    chatbot=gr.Chatbot(type='messages',height=500,label='聊天机器人')  #指定空白页大小
    with gr.Row():
        #文字输入区域,按照占比来算，每个区域来书写
        with gr.Column(scale=4):
            user_input=gr.Textbox(placeholder='please input',label='文字输入',max_lines=5)

            submit_btn=gr.Button('发送',variant='primary')
        with gr.Column(scale=1):
            # 写一个按钮加语音输入
            audio_input=gr.Audio(sources=['microphone'],label='语音输入',type='filepath',format='wav')
    #回车发送提交文字功能
    chat_message=user_input.submit(add_message,[chatbot,user_input],[chatbot,user_input])
    chat_message.then(execute_chain,chatbot,chatbot)

    #语音输入框功能
    audio_input.change(read_audio,[audio_input],[user_input])


res=final_chain.invoke({"input":"你好，我是李浩","config":{"configurable":{"session_id":"ques1"}}},
                                           config={"configurable":{"session_id":"ques1"}})
res1=final_chain.invoke({"input":"我是谁","config":{"configurable":{"session_id":"ques1"}}},
                                           config={"configurable":{"session_id":"ques1"}})
res2=final_chain.invoke({"input":"我是谁","config":{"configurable":{"session_id":"ques1"}}},
                                           config={"configurable":{"session_id":"ques2"}})
print(res)
print(res1)
print(res2)