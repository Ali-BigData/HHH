import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Cement':12, 'Big-Gravel':29, 
'Flyash':6,'Water':44,'Superplasticizer':12,'Small-Gravel':37,'Sand':8,'Days':5})

print(r.json())


