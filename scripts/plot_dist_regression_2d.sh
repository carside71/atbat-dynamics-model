#!/bin/bash
# 回帰ターゲットの 2D 密度プロット（ヒートマップ値域設定の参考用）
# launch_angle x spray_angle は 2D ヒートマップの軸に対応

cd /workspace/atbat-dynamics-model

python -m tools.plot_distribution /workspace/datasets/statcast-customized-v2 \
    --plot-2d spray_angle:launch_angle launch_speed:hit_distance_sc \
    --filter-swing \
    --reg-target-filter all \
    --output-dir /workspace/outputs/distribution
