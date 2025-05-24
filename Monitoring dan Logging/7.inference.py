import requests

url = "http://localhost:5001/invocations"
headers = {"Content-Type": "application/json"}

data = {
    "dataframe_split": {
        "columns": [str(i) for i in range(1120)],  # "0" s.d. "1119"
        "data": [[0.5] * 1120]  # <- data dummy, bisa ambil dari 1 baris `zara_ready.csv` kecuali kolom terakhir
    }
}

response = requests.post(url, json=data, headers=headers)
print("Status:", response.status_code)
print("Result:", response.json())
