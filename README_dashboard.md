# Local O1 Web Dashboard

## Overview
A Streamlit-based web interface for submitting tasks, monitoring agent/system performance, browsing task history, and visualizing workflow execution for the Local O1 system.

## Features
- **Task submission**: Enter and run tasks directly from the browser
- **Real-time monitoring**: View agent performance, memory usage, and system resources (auto-refresh every 3s)
- **Workflow visualization**: See agent interactions and timing for each task
- **Task history**: Browse, filter, and view results of previous tasks
- **Vector memory integration**: Instantly see similar past tasks/results
- **Privacy controls**: Honors vector memory privacy settings

## Installation & Startup
```sh
zsh scripts/run_dashboard.sh
```
- Opens at http://localhost:8501 in your browser

> **Note:** If your browser does not open automatically, manually visit [http://localhost:8501](http://localhost:8501) after running the dashboard script.

## Configuration
- Edit `ui_dashboard_config.json` to change refresh interval, history size, privacy, and port

## Usage Example
1. Enter a task in the sidebar and click "Run Task"
2. View the result, workflow, and similar past tasks
3. Monitor system and agent stats in real time
4. Browse and filter task history

## API Endpoints
- (Planned) `/run_task` for programmatic task submission

## Screenshots
- ![Dashboard Screenshot](docs/dashboard_screenshot.png)

---

**This dashboard is designed for local use, fast feedback, and seamless integration with the Local O1 system.**
