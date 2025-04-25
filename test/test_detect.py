import requests

# Replace with the IP of your Flask server
url = "http://192.168.23.152:5000/detect"

with open("test.jpeg", "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Detection Output:", response.text)
