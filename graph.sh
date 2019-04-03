#!/bin/sh
echo "Index,Time,TimePerElement" > b.csv
./cmake-build-debug/ocltest | awk '{print $2 ","$4 "," $6}' >> b.csv
./graph.py
viewnior c.png
