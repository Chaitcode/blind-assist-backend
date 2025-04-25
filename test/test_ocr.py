import requests

with open("ocr.jpg", "rb") as f:
    res = requests.post("http://192.168.23.152:5000/ocr", files={"image": f})
    print("Status Code:", res.status_code)
    print("OCR Output:", res.text)
