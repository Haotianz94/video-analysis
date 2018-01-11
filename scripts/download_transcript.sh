#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "$line"
    ia download $line --glob "*.srt"
#     mv $line/$line.cc5.srt .
#     rm -rf $line

done < "$1"