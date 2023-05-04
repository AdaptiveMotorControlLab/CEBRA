#!/bin/bash
# Identify binary files in the repository:

list_files() {
    git ls-tree --full-tree --name-only -r HEAD \
    | xargs -n1 grep -IL . \
    | xargs -n1 wc -c \
    | grep -v '^0 '
}

NUM_FILES=$(list_files | wc -l)
if [[ ${NUM_FILES} -ne 0 ]]; then
    echo "Repository contains binary files:"
    list_files
    exit 1
fi
