#!/bin/bash
file_name=$1
echo "subject,assignmentid,ord,time,event,instance"
grep event ${file_name} | sed 's/\"\"\"//g' | awk '{print $1""$3","$7}' | sed 's/event:\[//g' | sed 's/instance:\[//g' | sed 's/\]//g' | sed 's/:/,/g'
