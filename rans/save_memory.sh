#!/bin/bash

WATCH_DIR_1="initial_rans_runs"
WATCH_DIR_2="restart_rans_runs"

while true; do
    FILES=$(find "$WATCH_DIR_1" "$WATCH_DIR_2" -type f -name "*f00001*")
    for file in $FILES; do
        echo $file
        [ -f "$file" ] && rm -f "$file"
    done
    sleep 5
done
