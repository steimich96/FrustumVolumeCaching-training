#!/bin/bash

MIP360_DATA_ROOT="../data/360_v2/"
DATASETS=(
	# "bonsai" \
	# "counter" \
	# "kitchen" \
	# "room" \
	# "bicycle" \
	"garden" \
	# "stump" \
  # "flowers" \
  # "treehill" \
)

FIXED_HYPER_PARAMS="--max_steps 60000 --unbounded --gradient_scaling --distortion_loss_factor=1e-2 --sh_small_degree=4"
FIXED_VIEWDEP_HYPER_PARAMS="--viewdep_train --not_density_in_latents"

HYPER_PARAMS=( \
  # "--max_steps 60000" \
  # "${FIXED_HYPER_PARAMS} --zinterpol_train" \
  # "${FIXED_HYPER_PARAMS} --zinterpol_train --n_latents 8 --n_mlp_base_outputs 8" \
  "${FIXED_HYPER_PARAMS} ${FIXED_VIEWDEP_HYPER_PARAMS} --zinterpol_train --n_viewdep_samples=4 --viewdep_max_cone_angle=25.0 --n_latents 8 --mlp_head_n_layers 1 --mlp_head_n_neurons 128 --use_freq_encoding" \
  # "${FIXED_HYPER_PARAMS} ${FIXED_VIEWDEP_HYPER_PARAMS} --zinterpol_train --n_viewdep_samples=4 --viewdep_max_cone_angle=25.0 --n_latents 8 --mlp_head_n_layers 1 --mlp_head_n_neurons 128 --use_freq_encoding --gridenc_n_levels=12 --gridenc_max_resolution=8192 --log2_hashmap_size=22" \
  # "${FIXED_HYPER_PARAMS} ${FIXED_VIEWDEP_HYPER_PARAMS} --zinterpol_train --n_viewdep_samples=4 --viewdep_max_cone_angle=25.0 --n_latents 4 --mlp_head_n_layers 1 --mlp_head_n_neurons 128 --use_freq_encoding" \
)

EXPERIMENT_NAMES=( \
  # "ingp_big" \
  # "ingp_ours" \
  # "ingp_ours_latent_8" \
  "ours" \
  # "ours_huge" \
  # "ours_latent_4" \
)

idx=0
for hyper_params in "${HYPER_PARAMS[@]}"; do
  for dataset in "${DATASETS[@]}"; do

    timestamp=$(date "+%Y-%m-%d_%H%M%S")

    echo "Launched ${dataset}, ${timestamp}, ${EXPERIMENT_NAMES[$idx]}, ${idx}"
    mkdir -p "output/${dataset}/${EXPERIMENT_NAMES[$idx]}"

    python train.py \
      --scene "${dataset}" \
      --data_root "${MIP360_DATA_ROOT}" \
      --export_model --export_dir "output/${dataset}" \
      --save_model --save_dir "output/${dataset}" \
      --experiment "${EXPERIMENT_NAMES[$idx]}" ${hyper_params} # > "output/${dataset}/${EXPERIMENT_NAMES[$idx]}/log.txt"

    python eval.py \
      --scene "${dataset}" \
      --data_root "${MIP360_DATA_ROOT}" \
      --export_model --export_dir "output/${dataset}" \
      --checkpoint "output/${dataset}/${EXPERIMENT_NAMES[$idx]}/model.ckpt" \
      --eval_viewdep --eval_viewdep_max_cone_angle 45.0 --eval_viewdep_n_samples 4 \
      --experiment "${EXPERIMENT_NAMES[$idx]}" ${hyper_params}  # > "output/${dataset}/${EXPERIMENT_NAMES[$idx]}/rot_eval.txt"
  done
  idx=$((idx + 1))
done