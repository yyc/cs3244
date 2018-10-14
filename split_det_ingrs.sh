#!/bin/bash
# @500poundbear

FILENAME="det_ingrs.json"
WC_FILE="361085654" # TODO: run wc command instead

CHARS_PER_CHUNK="10000"
NUM_CHUNKS=$((($WC_FILE+($CHARS_PER_CHUNK-1))/$CHARS_PER_CHUNK)) # Hack to round up

echo $NUM_CHUNKS

NUM_CHUNKS=10

cnt=0
while [ $cnt -le $NUM_CHUNKS ]
do
    starting_ind=$(($cnt * $CHARS_PER_CHUNK + 1))
    ending_ind=$((($cnt+1) * $CHARS_PER_CHUNK))
    filename="_"
    echo "$starting_ind to $ending_ind"
    cat $FILENAME | cut -c $starting_ind-$ending_ind > det_ingrs_trunc_$starting_ind_$ending_ind.json    
    cnt=$(($cnt + 1))
done

echo Completed
