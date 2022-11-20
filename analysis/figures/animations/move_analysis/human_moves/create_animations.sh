#!/bin/bash
cd "$1"
for dir in */
do
    dirr=${dir%*/}
    subj=${dirr##*/}
    if [[ ${subj::1} != "A" ]]
    then
        cd "$dir"
        echo "$subj"
        convert "-delay" "50" "-loop" "0" "*.svg" "$1_$subj.gif"
        cd ".."
    fi
done
