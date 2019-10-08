import requests

url = 'http://localhost:3000/predict'

r = requests.post(url,json={'text': '-475 60 8 6221.92 6178.23 0.530438 0.336245 2238.601188'})
print(r.json())