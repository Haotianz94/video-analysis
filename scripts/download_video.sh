#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "$line"
    if [ ! -e "../data/videos/$line.mp4" ]; then
        gsutil cp gs://esper/tvnews/videos/$line.mp4 ../data/videos/
    fi
    
#     ia download $line --glob "*.srt"
#     if [ -e "$line/$line.cc5.srt" ]; then
#         mv $line/$line.cc5.srt ../data/videos/
#         echo "find $line.cc5.srt"
#     elif [ -e "$line/$line.cc1.srt" ]; then
#         mv $line/$line.cc1.srt ../data/videos/
#         echo "find $line.cc1.srt"
#     else
#         echo "$line does not have transcript!!!"
#     fi 
#     rm -rf $line

done < "$1"