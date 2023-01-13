#!/usr/bin/env bash
python3.11 main.py -p eil51 -m opt -td -opt gradient -oc 60 -i 6 &>> output.log &
python3.11 main.py -p berlin52 -m opt -td -opt gradient -oc 60 -i 6 &>> output.log &
wait
python3.11 main.py -p pr136 -m opt -td -opt gradient -oc 60 -i 6 &> output.log 
wait
python3.11 main.py -p pr226 -m opt -td -opt gradient -oc 60 -i 6 &> output.log 
wait
python3.11 main.py -p d198 -m opt -td -opt gradient -oc 60 -i 6 &> output.log 