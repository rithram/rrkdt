#!/bin/bash

data_file=$1
data_type=$2
output_prefix=$3

ks=(10 100)
muls=(1 2)
ntrees=50

echo "Data set: $data_file (type: $data_type)"

for k in ${ks[@]}; do
    echo "->Investigating $k-nearest-neighbors problem ..."
    for mul in ${muls[@]}; do
        leaf_size=$(echo "$k * $mul" | bc)
        echo "--> Running experiment with { -N : $leaf_size, -T : $ntrees } ..."
        prefix="$output_prefix.k$k.N$leaf_size.T$ntrees"
        rfile="$prefix.results"
        ffile="$prefix.figures.png"

        python2.7 expt.py \
                  -d $data_file \
                  -t $data_type \
                  -f 0.1 \
                  -k $k \
                  -N $leaf_size \
                  -T $ntrees \
                  -r $rfile \
                  -g $ffile
        echo "-->Results in $rfile and $ffile"
    done
done

echo "Output written in files prefixed with '$output_prefix'"
