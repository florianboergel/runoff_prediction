#!/bin/bash

# Search space definition
HIDDEN_DIMS=(4 6 8)
KERNEL_SIZES=("(3,3)" "(5,5)" "(7,7)")
INPUT_SIZES=(25 30 35)
NUM_LAYERS=(1 2)

# Iterate over the hyperparameters
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for kernel_size in "${KERNEL_SIZES[@]}"; do
        for input_size in "${INPUT_SIZES[@]}"; do
            for num_layer in "${NUM_LAYERS[@]}"; do

                # Construct a meaningful model name
                model_name="Model_H${hidden_dim}_K${kernel_size//,/x}_I${input_size}_L${num_layer}"

                # Construct the command
                cmd="python testNewModelArchitecture.py --modelName $model_name --hidden_dim $hidden_dim --kernel_size $kernel_size --input_size $input_size --num_layers $num_layer --num_epochs 40"

                # Execute the command
                echo "Executing: $cmd"
                $cmd

            done
        done
    done
done
