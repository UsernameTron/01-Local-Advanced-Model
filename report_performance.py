import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load benchmark results
with open('benchmark_results.json') as f:
    bench = json.load(f)

with open('parameter_search_results.json') as f:
    param_results = json.load(f)

# Plot performance comparison
plt.figure(figsize=(8,5))
plt.plot(bench['distilled_times'], label='Distilled')
plt.plot(bench['original_times'], label='Original')
plt.xlabel('Step')
plt.ylabel('Response Time (s)')
plt.title('Executor Response Times')
plt.legend()
plt.tight_layout()
plt.savefig('performance_comparison.png')

# Plot parameter search results
plt.figure(figsize=(8,5))
params = [str(r['params']) for r in param_results]
times = [r['avg_time'] for r in param_results]
plt.barh(params, times)
plt.xlabel('Avg Response Time (s)')
plt.title('Parameter Search Results')
plt.tight_layout()
plt.savefig('parameter_search.png')

# Save HTML report
html = f"""
<html><head><title>Local O1 Performance Report</title></head><body>
<h1>Local O1 Performance Report ({datetime.now().isoformat()})</h1>
<h2>Executor Response Times</h2>
<img src='performance_comparison.png' width='600'/>
<h2>Parameter Search Results</h2>
<img src='parameter_search.png' width='600'/>
</body></html>
"""
with open('performance_report.html', 'w') as f:
    f.write(html)

# Maintain history
history_file = 'performance_history.json'
if os.path.exists(history_file):
    with open(history_file) as f:
        history = json.load(f)
else:
    history = []
history.append({
    'timestamp': datetime.now().isoformat(),
    'bench': bench,
    'param_results': param_results
})
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)

print('Performance report generated: performance_report.html')
