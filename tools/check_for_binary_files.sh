#!/bin/bash
# Identify binary files in the repository:

exit $(
    git ls-tree --full-tree --name-only -r HEAD \
    | xargs -n1 grep -IL . \
    | xargs -n1 wc -c \
    | grep -v '^0 ' \
    | tee /dev/tty \
    | wc -l
)
