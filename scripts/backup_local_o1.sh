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
