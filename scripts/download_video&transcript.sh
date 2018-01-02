#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
#     srt=${line/mp4/cc5.srt}
    echo "gs://esper/$srt"
    gsutil cp gs://esper/tvnews/videos/$line.mp4 ../data/videos/
#     gsutil cp gs://esper/tvnews/videos/$line.cc5.srt ../data/videos/    

done < "$1"