from prometheus_client import Gauge, start_http_server
import psutil
import time

# Define metrics
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
cpu_temp = Gauge('cpu_temperature_celsius', 'CPU temperature (simulated)')
cpu_mem = Gauge('memory_usage_percent', 'Memory usage percentage')

# Start Prometheus HTTP server
start_http_server(8000)
print("✅ CPU metrics available at http://localhost:8000/metrics")

while True:
    # System metrics
    cpu_usage.set(psutil.cpu_percent())
    cpu_mem.set(psutil.virtual_memory().percent)
    
    # Simulated temperature (since macOS doesn’t expose real temps easily)
    cpu_temp.set(40 + psutil.cpu_percent() * 0.5)
    
    time.sleep(2)
