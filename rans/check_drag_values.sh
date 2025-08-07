#!/bin/bash

# Script to check for significant changes in drag_px values
# Usage: ./check_drag_changes.sh

THRESHOLD=0.0001
BASE_DIR="restart_rans_runs"

# Function to check drag_px changes in a single drag.txt file
check_drag_changes() {
    local file="$1"
    local path="$2"
    
    # Extract the last two drag_px values from the file
    # Look for lines containing "drag_px" and get the last two
    local drag_values=$(grep "drag_px" "$file" | tail -2 | awk '{print $3}')
    
    # Convert to array
    local values_array=($drag_values)
    
    # Check if we have at least 2 values
    if [ ${#values_array[@]} -lt 2 ]; then
        return 1
    fi
    
    local second_last="${values_array[0]}"
    local last="${values_array[1]}"
    
    # Calculate absolute difference using awk (bash doesn't handle floating point well)
    local diff=$(awk -v a="$second_last" -v b="$last" 'BEGIN {
        diff = a - b
        if (diff < 0) diff = -diff
        print diff
    }')
    
    # Compare with threshold
    local significant=$(awk -v diff="$diff" -v thresh="$THRESHOLD" 'BEGIN {
        if (diff > thresh) print "1"
        else print "0"
    }')
    
    if [ "$significant" = "1" ]; then
        echo "$path"
        return 0
    fi
    
    return 1
}

# Main script
echo "Checking for significant drag_px changes (threshold: $THRESHOLD)"
echo "=================================================="

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found!"
    exit 1
fi

found_changes=0

# Traverse the directory structure: restart_rans_runs/dir/subdir/subsubdir/drag.txt
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        
        for subdir in "$dir"/*; do
            if [ -d "$subdir" ]; then
                subdir_name=$(basename "$subdir")
                
                for subsubdir in "$subdir"/*; do
                    if [ -d "$subsubdir" ]; then
                        subsubdir_name=$(basename "$subsubdir")
                        drag_file="$subsubdir/drag.txt"
                        
                        if [ -f "$drag_file" ]; then
                            path_string="$dir_name/$subdir_name/$subsubdir_name"
                            
                            if check_drag_changes "$drag_file" "$path_string"; then
                                found_changes=1
                            fi
                        fi
                    fi
                done
            fi
        done
    fi
done

if [ $found_changes -eq 0 ]; then
    echo "No significant drag_px changes found."
fi

echo "=================================================="
echo "Check complete."
