#!/bin/sh
echo "Index,Time,TimePerElement" > b.csv
./ocltest | tail -n +30 |  awk '{print $2 ","$4 "," $6}' >> b.csv
./graph.py
viewnior c.png
