#!/bin/bash
# 投球特徴量の 1D 分布（フィルタなし、train split 全データ）

cd /workspace/atbat-dynamics-model

python -m tools.plot_distribution /workspace/datasets/statcast-customized-v2 \
    --columns release_speed release_spin_rate pfx_x pfx_z plate_x plate_z \
    --output-dir /workspace/outputs/distribution
