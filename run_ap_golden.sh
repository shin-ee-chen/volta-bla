#! /bin/bash
sizes=(41)
groups=(1 2 3)
seeds=(0 42 1042)

source /root/miniconda3/bin/activate /root/autodl-tmp/conda_env/volta

for size in ${sizes[@]}
do
    for group in ${groups[@]}
    do
        for seed in ${seeds[@]}
        do
            cmd='python /root/autodl-tmp/volta/train_concap_modify.py --tasks_config_file /root/autodl-tmp/volta/exmaple_xinyi_bla_train/task_configs/ctrl_active_tasks_golden_'${size}'_'${group}'.yml --seed '${seed}
            echo ${cmd}
            eval ${cmd}
			
			cmd=${cmd}' --freeze_before_layer 34'
			echo ${cmd}
            eval ${cmd}
        done
    done
done
