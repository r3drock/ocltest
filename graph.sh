#!/bin/sh
echo "Index,Time,TimePerElement,Index4,Time4,Time4PerElement" > b.csv
./ocltest | tail -n +29 |  awk '{print $2 ","$4 "," $6 "," $8 "," $10 "," $12}' >> b.csv
./graph.py
viewnior c.png
