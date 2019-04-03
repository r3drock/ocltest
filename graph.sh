#!/bin/sh
echo "Index,Time" > b.csv
./a.out | awk '{print $2 "," $6}' >> b.csv
./graph.py
viewnior c.png
