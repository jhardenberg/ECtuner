#!/bin/bash

mkdir -p tuneout
# Special case
fn="s000"
python ectuner.py $fn 1990 1997 -o tuneout/tune-$fn.yml -i 0.2 -p 30 -l WARNING

for a in {0..9}; do
    for b in {1..2}; do
        fn="s0${a}${b}"
	echo Optimizing $fn
        python ectuner.py $fn 1990 1997 -o tuneout/tune-$fn.yml -i 0.2 -p 30 -l WARNING
    done
done
