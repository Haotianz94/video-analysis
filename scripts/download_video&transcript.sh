#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "$line"
    gsutil cp gs://esper/tvnews/videos/$line.mp4 ../data/videos/
    gsutil cp gs://esper/tvnews/videos/$line.cc5.srt ../data/videos/    

done < "$1"