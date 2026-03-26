#!/bin/bash
# 回帰ターゲット 4 列の 1D ヒストグラム（regression scope フィルタ付き）
# ヒートマップの heatmap_range_* 設定の参考に使用

cd /workspace/atbat-dynamics-model

python -m tools.plot_distribution /workspace/datasets/statcast-customized-v2 \
    --columns launch_speed launch_angle hit_distance_sc spray_angle \
    --filter-swing \
    --reg-target-filter all \
    --output-dir /workspace/outputs/distribution
