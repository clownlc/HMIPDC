#!/bin/bash  
sc=10.0
for name in 'acm' 'dblp' 'cite' 'amap' 'cora'

do
    for lr in 0.01

    do
        for clu_loss in 1

        do
            for pn_loss in 0.01

            do
                for max_positive in 0.5 0.6 0.7 0.8 0.9

                do
                    for min_negative in 0.1 0.2 0.3 0.4 0.5

                    do
                        for input_dim in 50 100 150 200

                        do

                            echo "--name $name --pn_loss $pn_loss --max_positive $max_positive --min_negative $min_negative --input_dim $input_dim"
                            python daegc.py --name $name --pn_loss $pn_loss --max_positive $max_positive --min_negative $min_negative --input_dim $input_dim

                        done
                    done
                done
            done
        done
    done
done
