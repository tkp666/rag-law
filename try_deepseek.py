# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(api_key='sk-c18968b606ea421fa992a6cda26c7111', base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个专业的法律顾问，用中文回答法律相关问题"},
        {"role": "user", "content": "劳动合同纠纷应该怎么处理？"},
    ],
    stream=False
)

print(response.choices[0].message.content)