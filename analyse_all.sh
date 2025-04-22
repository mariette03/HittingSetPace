#!/bin/bash

for folder in "ds"; do
    echo "Files in $folder:"
    for file in instances/$folder/exact/*; do
        if [[ -f "$file" ]]; then
            target/release/hitdomsolver analysis $file
        fi
    done
done
