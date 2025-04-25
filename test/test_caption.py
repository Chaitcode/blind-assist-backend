import requests

with open("test.jpeg", "rb") as f:
    res = requests.post("http://192.168.23.152:5000/caption", files={"image": f})

print(res.json())
