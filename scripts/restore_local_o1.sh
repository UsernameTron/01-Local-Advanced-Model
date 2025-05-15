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
