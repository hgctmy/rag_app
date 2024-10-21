import requests
import json

post_url = "http://127.0.0.1:8000/generate-response"  # 自分のアドレスを入力
param = {
    "session_key": "test",
    "model": "gpt-4o-mini",
    "user_input": "行いたい質問"
}  # 条件を入力


# POST送信
response = requests.post(
    post_url,
    json=param,
)
print(response.json())
