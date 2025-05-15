# Local O1 Production Deployment Guide

This guide provides step-by-step, executable instructions for deploying the optimized Local O1 system in production on both M-series Macs and Linux servers. It includes setup scripts, systemd service definitions, backup/restore strategies, performance tuning, security best practices, and troubleshooting.

---

## 1. System Requirements & Dependencies

### Hardware
- **CPU:** Apple Silicon (M1/M2/M3) or x86_64 Linux (8+ cores recommended)
- **RAM:** 16GB minimum (32GB+ recommended for large workloads)
- **Disk:** SSD, 20GB+ free
- **GPU:** Optional, for advanced model acceleration

### Software
- macOS 13+ or Ubuntu 22.04+ (or compatible Linux)
- Python 3.10+
- Ollama (for LLM agent execution)
- Docker (optional, for containerized deployment)

---

## 2. Automated Setup Script

Save as `scripts/setup_local_o1.sh` and run with `zsh scripts/setup_local_o1.sh`.

```zsh
#!/bin/zsh
set -e

# 1. Python & venv
if ! command -v python3 &>/dev/null; then
  echo "Python3 not found. Install Python 3.10+ first."; exit 1
fi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Core dependencies
pip install -r requirements.txt || pip install streamlit psutil faiss-cpu torch transformers sentence-transformers requests

# 3. Ollama
if ! command -v ollama &>/dev/null; then
  echo "Ollama not found. Installing..."
  if [[ "$(uname)" == "Darwin" ]]; then
    brew install ollama
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi
ollama serve &
ollama pull phi3.5

# 4. Create default benchmark file if missing
python scripts/create_benchmark_results.py

# 5. Permissions
chmod +x scripts/*.sh

# 6. (Optional) Docker build
# docker build -t local-o1 .

echo "[OK] Local O1 setup complete."
```

---

## 3. Deployment Checklist

- [ ] Hardware meets requirements
- [ ] Python 3.10+ and venv created
- [ ] All Python dependencies installed
- [ ] Ollama installed, running, and model pulled
- [ ] `benchmark_results.json` present
- [ ] Vector memory files (`vector_memory.index`, `vector_memory_meta.json`) backed up
- [ ] Systemd service enabled (see below)
- [ ] Firewall configured (see Security)

---

## 4. Systemd Service Definitions

### `local_o1_dashboard.service`

Save as `/etc/systemd/system/local_o1_dashboard.service`:

```ini
[Unit]
Description=Local O1 Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/local_o1
ExecStart=/path/to/local_o1/venv/bin/streamlit run ui_dashboard.py --server.port 8501 --server.headless false
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### `ollama.service` (if not managed by default)

```ini
[Unit]
Description=Ollama LLM Service
After=network.target

[Service]
Type=simple
User=YOUR_USER
ExecStart=ollama serve
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## 5. Backup & Restore Strategy

### Backup
```zsh
#!/bin/zsh
# scripts/backup_local_o1.sh
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=backup_$DATE
mkdir -p $BACKUP_DIR
cp vector_memory.index vector_memory_meta.json $BACKUP_DIR/
cp -r output/ logs/ $BACKUP_DIR/
cp benchmark_results.json $BACKUP_DIR/
cp -r dataset/ $BACKUP_DIR/
echo "Backup complete: $BACKUP_DIR"
```

### Restore
```zsh
#!/bin/zsh
# scripts/restore_local_o1.sh
set -e
BACKUP_DIR=$1
if [[ ! -d $BACKUP_DIR ]]; then echo "Backup dir not found"; exit 1; fi
cp $BACKUP_DIR/vector_memory.index $BACKUP_DIR/vector_memory_meta.json .
cp -r $BACKUP_DIR/output $BACKUP_DIR/logs .
cp $BACKUP_DIR/benchmark_results.json .
cp -r $BACKUP_DIR/dataset .
echo "Restore complete."
```

---

## 6. Performance Tuning

- **M-series Mac:**
  - Use `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon acceleration
  - Monitor RAM/CPU via dashboard
- **Linux (x86_64):**
  - Use CUDA if available (`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`)
  - Increase ulimit for file handles if running many agents
- **General:**
  - Prune vector memory regularly
  - Use SSD for vector store
  - Adjust `orchestration_config.json` for max agents/executors

---

## 7. Security Best Practices

- Run services as non-root users
- Restrict dashboard port (8501) to trusted networks
- Use firewall (ufw, pf, etc.) to block unwanted access
- Regularly update Python packages and Ollama
- Protect backup files with proper permissions
- Review logs for suspicious activity
- Never expose Ollama or dashboard to the public internet without authentication

---

## 8. Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| Dashboard won't start | Check Python/venv, run `zsh scripts/setup_local_o1.sh` |
| Ollama connection errors | Ensure `ollama serve` is running, check firewall |
| Model not found | Run `ollama pull phi3.5` |
| Vector memory errors | Restore from backup, check disk space |
| High RAM/CPU | Reduce agent count, prune memory, upgrade hardware |
| Permission denied | `chmod +x scripts/*.sh` and check user rights |
| Systemd not starting | `journalctl -u local_o1_dashboard` for logs |

---

## 9. (Optional) Docker Deployment

Add a `Dockerfile` and `docker-compose.yml` for containerized setups. See [README_dashboard.md] for examples.

---

## 10. Benchmark Expectations

| Hardware | Task | Latency (s) | RAM (GB) |
|----------|------|-------------|----------|
| M1/M2 Mac | Summarize doc | 8-15 | 4-8 |
| x86_64 8-core | Summarize doc | 10-20 | 8-16 |
| M1/M2 Mac | Vector search | <1 | 2 |

---

**For further help, see the project READMEs or contact the maintainer.**
