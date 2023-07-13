#!/bin/bash
sc=10.0
for name in 'acm' 'cite' 'dblp' 'amap'

do
    for i in $(seq 1 5)

    do

          echo  "--i $i --name $name"
          python daegc.py --name $name

    done

done


#
#for i in $(seq 0 10)
#do
#echo "--seed $(expr $i \* 10)"
#
#done