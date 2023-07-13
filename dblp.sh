#!/bin/bash  
sc=10.0
for name in 'dblp'

do
    for lr in 0.01

    do
        for clu_loss in 1

        do
            for pn_loss in 0.01

            do
                for max_positive in 0.7

                do
                    for min_negative in 0.2

                    do
                        for input_dim in 100

                        do
                            for i in $(seq 1 20)

                            do

                                echo "--i $i --name $name --pn_loss $pn_loss --max_positive $max_positive --min_negative $min_negative --input_dim $input_dim"
                                python daegc.py --name $name --i $i --pn_loss $pn_loss --max_positive $max_positive --min_negative $min_negative --input_dim $input_dim

                            done
                        done
                    done
                done
            done
        done
    done
done
