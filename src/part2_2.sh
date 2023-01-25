#!/usr/bin/env bash
python3.11 main.py -p pr226 -m opt -di 0.1 -opt gradient -oc 60 -i 6 &> output.log &
python3.11 main.py -p pr226 -m opt -di 0.25 -opt gradient -oc 60 -i 6 &> output.log &
python3.11 main.py -p pr226 -m opt -di 0.5 -opt gradient -oc 60 -i 6 &> output.log &