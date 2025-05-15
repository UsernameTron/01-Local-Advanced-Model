# Local O1 Monitoring Dashboard Setup

## Overview
This dashboard tracks key performance metrics for the Local O1 system, including executor response times, pipeline latency, model quality, and parameter optimization trends.

## Metrics Tracked
- Executor response time (distilled/original)
- Pipeline total time
- Quality validation pass rate
- Parameter optimization results
- Historical trends (from `performance_history.json`)

## Setup Instructions
1. **Install requirements:**
   - `pip install matplotlib flask` (for a simple local dashboard)
2. **Run the dashboard server:**
   - Use a Flask app to serve `performance_report.html` and visualizations
   - Example:
     ```python
     from flask import Flask, send_file
     app = Flask(__name__)
     @app.route('/')
     def report():
         return send_file('performance_report.html')
     if __name__ == '__main__':
         app.run(port=8080)
     ```
3. **Access the dashboard:**
   - Open [http://localhost:8080](http://localhost:8080) in your browser
4. **Customize:**
   - Edit `monitoring_dashboard_config.json` to add/remove metrics or change refresh interval

## Automation
- The dashboard is updated automatically by the CI/CD pipeline after each run.
- Historical data is appended to `performance_history.json` for trend analysis.

---

**This setup provides continuous visibility into Local O1 system performance and optimization.**
