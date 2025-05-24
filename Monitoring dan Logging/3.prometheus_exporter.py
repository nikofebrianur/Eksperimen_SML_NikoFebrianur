from prometheus_client import start_http_server, Summary, Gauge
import time
import random

# Metrik yang akan dipantau
inference_latency = Summary("inference_latency_seconds", "Waktu proses inference")
model_mse = Gauge("model_mse", "Nilai MSE model dari evaluasi terakhir")
prediction_output = Gauge("last_prediction_value", "Nilai prediksi terakhir yang diberikan oleh model")

# Fungsi simulasi inference
@inference_latency.time()
def simulate_inference():
    # Simulasi latency & prediksi
    time.sleep(random.uniform(0.2, 0.8))
    mse = random.uniform(200000, 600000)
    pred = random.uniform(1500, 3000)

    # Logging ke metrik
    model_mse.set(mse)
    prediction_output.set(pred)

    print(f"[Simulasi] MSE: {mse:.2f} | Prediksi: {pred:.2f}")

if __name__ == "__main__":
    print("Menyalakan Prometheus Exporter di port 8000...")
    start_http_server(8000)  # bisa diakses Prometheus di localhost:8000/metrics

    while True:
        simulate_inference()
        time.sleep(5)  # Update metrik setiap 5 detik
