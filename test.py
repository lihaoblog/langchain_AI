from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:6006/v1",api_key="XXX")

resp=client.chat.completions.create(
    model='qwen3-8b',
    messages=[{'role':'user','content':'介绍一下什么是深度学习，简单一句话回答'}],
    #发散度
    temperature=0.8,
    #保证生成得到内容不重复
    presence_penalty=1.5,
    #千问3特有参数,开启深度思考
    extra_body={'chat_template_kwargs':{'enable_thing':True}},
)
print(resp.choices[0].message.content)
print(resp)