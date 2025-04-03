import requests

url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}

messages = [
    {"role": "system", "content": """Ты умный помощник, говори кратко и по делу."""},
    {"role": "user", "content": "Как работает квантовая механика?"}
]

data = {
    "model": "cognitivecomputations_dolphin3.0-r1-mistral-24b@iq3_xs",
    "messages": messages,
    "temperature": 0.1,
    "max_tokens": -1,
    "stream": False,
}

response = requests.post(url, json=data, headers=headers)


def get_text(response) -> str:
    return response.json()["choices"][0]["message"]["content"]


reply = get_text(response)
print("'''")
print(reply)
print("'''")

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "Очень интересно, продолжай"})
messages.append({"role": "assistant", "content": "На самом деле я терпеть не могу теоретическую физику! Потому что"})

response2 = requests.post(url, json=data, headers=headers)
print("'''")
print(messages[-1]["content"] + get_text(response2))
print("'''")
