#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "$line"
    if [ -e "../data/transcripts_raw2/$line/$line.cc5.srt" ]; then
        mv ../data/transcripts_raw2/$line/$line.cc5.srt ../data/transcripts/
        echo "find $line.cc5.srt"
    elif [ -e "../data/transcripts_raw2/$line/$line.cc1.srt" ]; then
        mv ../data/transcripts_raw2/$line/$line.cc1.srt ../data/transcripts/
        echo "find $line.cc1.srt"
    elif [ -e "../data/transcripts_raw2/$line/$line.align.srt" ]; then
        mv ../data/transcripts_raw2/$line/$line.align.srt ../data/transcripts/
        echo "find $line.align.srt"
    else    
        echo "$line does not have transcript!!!"
    fi

done < "$1"