#!/bin/bash

# search for all processes matching the given Python script name
PIDS=$(ps aux | grep "python -u $1" | grep -v "grep" | awk '{print $2}')

# kill these processes
for pid in $PIDS; do
	    echo "Killing process with PID: $pid"
	        kill -9 $pid
	done
