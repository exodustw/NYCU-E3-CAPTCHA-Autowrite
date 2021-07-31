import requests

url = 'http://127.0.0.1:5000/e3autologin'
files = {'file': open('0009.png', 'rb')}

rq = requests.post(url, files=files)

print(rq.text)
