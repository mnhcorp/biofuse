#!/bin/bash

for config in sweep-configs/oracle-single/sweep_config_*.yaml; do
    sweep_output=$(wandb sweep -p oracle-single-0725 "$config")
    agent_cmd=$(echo "$sweep_output" | grep -o 'wandb agent [^"]*')
    if [ -n "$agent_cmd" ]; then
        echo "$agent_cmd"
    fi
done