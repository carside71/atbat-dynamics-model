# src/models/

モデルアーキテクチャの実装を配置するパッケージ。
レジストリパターンにより、新しいモデルを追加するだけで学習・評価パイプラインから利用できます。

## 登録済みモデル一覧

| 名前 | ファイル | 説明 |
|------|----------|------|
| `atbat_dnn` | `atbat_dnn.py` | 共有バックボーン + 4 ヘッドの MLP (ReLU + BatchNorm) |
| `atbat_dnn_mdn` | `atbat_dnn_mdn.py` | 分類ヘッドは同一、回帰ヘッドを MDN (Mixture Density Network) に置換 |
| `atbat_resdnn` | `atbat_resdnn.py` | 残差接続 + GELU + LayerNorm でバックボーンを強化 |
| `atbat_resdnn_cascade` | `atbat_resdnn_cascade.py` | 上記 + カスケードヘッド（上流ヘッドの出力を下流に伝達） |
| `atbat_seq_resdnn` | `atbat_seq_resdnn.py` | 打席内系列エンコーダ (GRU/Transformer) + ResBlock バックボーン |
| `atbat_seq_resdnn_batter_hist` | `atbat_seq_resdnn_batter_hist.py` | 上記 + 階層 GRU 打者履歴エンコーダ |

---

## 共通構造

すべてのモデルは **埋め込み → バックボーン → マルチヘッド** の3段構成です。

```
入力
 ├─ カテゴリカル特徴量 ──→ [Embedding] ─┐
 ├─ 連続値特徴量 ──────────────────────┤──→ concat ──→ [Backbone] ──→ h
 └─ 順序特徴量 ────────────────────────┘                              │
                                                                    ├─→ swing_attempt  (B,)    logits
                                                                    ├─→ swing_result   (B, 9)  logits
                                                                    ├─→ bb_type        (B, 4)  logits
                                                                    └─→ regression     (B, 3)  values
```

### 出力（全モデル共通）

| キー | 形状 | 内容 |
|---|---|---|
| `swing_attempt` | `(B,)` | スイング試行 logit (binary) |
| `swing_result` | `(B, 9)` | スイング結果 logits (9 クラス) |
| `bb_type` | `(B, 4)` | 打球タイプ logits (4 クラス) |
| `regression` | `(B, 3)` | launch_speed, launch_angle, hit_distance_sc |

---

## 1. atbat_dnn

**シンプルな全結合ネットワーク。** ベースラインモデル。

```
Embedding concat
      │
      ▼
┌────────────────────┐
│ Linear(in, 256)    │
│ BatchNorm1d        │
│ ReLU               │╮
│ Dropout            ││ × backbone_num_layers (default 3)
└────────────────────┘│
      │ ◄─────────────╯
      ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ SA Head  │ │ SR Head  │ │ BT Head  │ │ Reg Head │
│ Lin→1    │ │ Lin→9    │ │ Lin→4    │ │ Lin→3    │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

- **バックボーン**: `Linear → BatchNorm1d → ReLU → Dropout` を `backbone_num_layers` 回スタック
- **ヘッド**: 各出力タスクに独立した `Linear` レイヤー
- **設定例**: `configs/dnn.yaml`

---

## 2. atbat_dnn_mdn

**DNN + 混合密度ネットワーク (MDN)。** 回帰ヘッドのみ MDN に置き換え。

```
Embedding concat
      │
      ▼
┌────────────────────┐
│ Linear → BN → ReLU │ × backbone_num_layers
│ → Dropout          │
└────────────────────┘
      │
      ├─→ SA Head  (Linear → 1)
      ├─→ SR Head  (Linear → 9)
      ├─→ BT Head  (Linear → 4)
      │
      └─→ MDN Head
           ├─→ π (mixing coefficients)  : Linear → K → Softmax
           ├─→ μ (means)                : Linear → K × 3
           └─→ σ (std deviations)       : Linear → K × 3 → ELU+1+ε
```

- **MDN ヘッド**: K 個のガウス分布の混合で回帰ターゲットをモデル化
- **推論時**: 最大重み成分の μ を予測値として採用
- **設定例**: `configs/dnn_mdn.yaml`（`mdn_num_components` で K を指定）

---

## 3. atbat_resdnn

**残差接続付き DNN。** ResBlock と ProjectedResBlock で勾配流を安定化。

```
Embedding concat ──→ Input Projection (Linear → LN → GELU)
      │
      ▼
┌─────────────────────────────────────────────┐
│              ResBlock / ProjectedResBlock   │
│  ┌─────────┐                                │
│  │ Input h │───────────────────┐ (skip)     │
│  └────┬────┘                   │            │
│       ▼                        │            │
│  Linear → LN → GELU → Dropout  │            │
│       ▼                        │            │
│  Linear → LN                   │            │
│       ▼                        ▼            │
│     h_out  ────────────────→  (+) ──→ GELU  │
│                                             │
│  ※ ProjectedResBlock: skip 側にも Linear→LN  │
└─────────────────────────────────────────────┘
      │  × backbone_num_layers
      ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ SA Head  │ │ SR Head  │ │ BT Head  │ │ Reg Head │
│Lin→GELU  │ │Lin→GELU  │ │Lin→GELU  │ │Lin→GELU  │
│→Lin→1    │ │→Lin→9    │ │→Lin→4    │ │→Lin→3    │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

- **入力投影**: 埋め込み結合後の次元を `backbone_hidden` に統一
- **ResBlock**: 次元が同一の場合。`Linear→LN→GELU→Dropout→Linear→LN + skip → GELU`
- **ProjectedResBlock**: 次元が異なる場合。skip 接続に射影 `Linear→LN` を挿入
- **ヘッド**: 2 層 MLP (`Linear→GELU→Linear`)
- **設定例**: `configs/resdnn.yaml`

---

## 4. atbat_resdnn_cascade

**カスケードヘッド付き ResBlock DNN。** ヘッド間に因果的依存関係を導入。

```
Embedding concat ──→ Input Projection ──→ ResBlock × N ──→ h
                                                           │
          ┌────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────┐
    │  SA Head   │──→ swing_attempt logit (sa)
    │ Lin→GELU   │
    │ →Lin→1     │
    └─────┬──────┘
          │ concat [h, sa]
          ▼
    ┌────────────┐
    │  SR Head   │──→ swing_result logits (sr)
    │ Lin→GELU   │
    │ →Lin→9     │
    └─────┬──────┘
          │ concat [h, sr]
          ▼
    ┌────────────┐
    │  BT Head   │──→ bb_type logits (bt)
    │ Lin→GELU   │
    │ →Lin→4     │
    └─────┬──────┘
          │ concat [h, bt]
          ▼
    ┌────────────┐
    │  Reg Head  │──→ regression (3)
    │ Lin→GELU   │
    │ →Lin→3     │
    └────────────┘
```

- **カスケードの流れ**: `h → SA → [h,sa] → SR → [h,sr] → BT → [h,bt] → Reg`
- **`detach_cascade`**: `True` にすると上流ヘッドからの勾配を `detach` し、下流ヘッド学習時に上流を更新しない
- **設計意図**: スイング試行 → スイング結果 → 打球タイプ → 回帰値 という野球の因果構造を反映
- **設定例**: `configs/resdnn_cascade.yaml`（`detach_cascade: true` / `resdnn_focal.yaml`）

---

## 5. atbat_seq_resdnn

**打席内系列エンコーダ付き ResBlock DNN。** 同一打席の過去投球情報を活用。

```
過去投球系列 (T 球分)                       現在の投球
─────────────────────                    ─────────────────
seq_pitch_type ──→ [Embed] ─┐            cat ──→ [Embed] ─┐
seq_swing_result → [Embed] ─┤            cont ────────────┤
seq_cont ───────────────────┤            ord ─────────────┘
seq_swing_attempt ──────────┘                    │
         │                                       │
    concat (T, D_seq)                    Embedding concat
         │                                       │
         ▼                                       │
┌────────────────┐                               │
│  系列エンコーダ  │                               │
│  GRU or        │                               │
│  Transformer   │                               │
└───────┬────────┘                               │
        │                                        │
        │ seq_embedding                          │
        └──────────────┬─────────────────────────┘
                       │
                    concat ──→ Projection
                                   │
                                   ▼
                             ResBlock × N
                                   │
                      ┌─────┬──────┼──────┐
                      ▼     ▼      ▼      ▼
                     SA    SR     BT    Reg
```

### 系列エンコーダの選択

#### GRU エンコーダ (`seq_encoder_type: gru`)

```
input (B, T, D_seq)  ──→ pack_padded_sequence (seq_mask で実長算出)
                              │
                              ▼
                         nn.GRU
                         (hidden_size = seq_hidden_dim)
                         (num_layers = seq_num_layers)
                         (bidirectional = seq_bidirectional)
                              │
                              ▼
                         h_n[-1] or cat(h_n[-2:])  ──→ seq_embedding
```

- 可変長系列に `pack_padded_sequence` で対応
- 双方向時は最終隠れ状態の forward/backward を concat
- **過去投球がない場合**（打席1球目）: ゼロベクトルを返却

#### Transformer エンコーダ (`seq_encoder_type: transformer`)

```
input (B, T, D_seq)  ──→ Linear(D_seq → seq_hidden_dim)
                              │
                              ▼
                    TransformerEncoderLayer × seq_num_layers
                    (nhead=4, dim_feedforward=seq_hidden_dim×4)
                    (src_key_padding_mask = ~seq_mask)
                              │
                              ▼
                    masked mean pooling ──→ seq_embedding
```

- `src_key_padding_mask` でパディング位置をマスク
- 出力をマスク付き平均プーリングで固定長ベクトル化
- **過去投球がない場合**: ゼロベクトルを返却

### 設定

```yaml
model:
  architecture: atbat_seq_resdnn
  max_seq_len: 10              # 過去投球の最大系列長
  seq_encoder_type: gru        # "gru" or "transformer"
  seq_hidden_dim: 64           # エンコーダ隠れ層次元
  seq_num_layers: 1            # エンコーダ層数
  seq_bidirectional: false     # GRU のみ: 双方向フラグ
```

- **設定例**: `configs/seq_resdnn.yaml`

---

## 6. atbat_seq_resdnn_batter_hist

**打者履歴エンコーダ付き系列 ResBlock DNN。** `atbat_seq_resdnn` を拡張し、打者の直近 N 打席（デフォルト50打席）の Statcast 全投球データを階層 GRU でエンコードして予測に利用する。「この打者は最近どのような投球にどう反応してきたか」という傾向をモデルに伝えることが目的。

### 全体構造

モデルは 3 つの独立したエンコーダの出力を結合して予測する。

```
═══════════════════════════════════════════════════════════════════════════════════
  (A) 打者履歴エンコーダ          (B) 打席内系列エンコーダ      (C) 現在の投球
═══════════════════════════════════════════════════════════════════════════════════

  過去 N 打席 × 各最大 P 球      同一打席の過去 T 球         カテゴリカル / 連続値 / 順序
  ──────────────────────       ────────────────         ─────────────────────

  [投球レベル: 各球ごと]         seq_pitch_type (T,)      pitch_type, stand,
  hist_pitch_type (N,P)        seq_cont (T,15)          batter, p_throws,
  hist_cont (N,P,15)           seq_swing_attempt (T,)   base_out_state, ...
  hist_swing_attempt (N,P)     seq_swing_result (T,)    release_speed, ...
  hist_swing_result (N,P)             │                        │
         │                            ▼                        ▼
         ▼                     ┌─────────────┐          ┌───────────┐
  ┌─────────────┐              │ GRU /       │          │ Embedding │
  │  Inner GRU  │              │ Transformer │          │  concat   │
  │  (B*N,P,D)  │              └──────┬──────┘          └─────┬─────┘
  └──────┬──────┘                     │                       │
         │ h_n[-1]                    │                       │
         ▼                            │                       │
  [打席レベル: 各打席ごと]               │                       │
  + bb_type_emb (打球種別)             │                       │
  + launch_speed (打球速度)            │                       │
  + launch_angle (打球角度)            │                       │
         │                            │                       │
         │ (B, N, D_ab)               │                       │
         ▼                            │                       │
  ┌─────────────┐                     │                       │
  │  Outer GRU  │                     │                       │
  │  (B,N,D_ab) │                     │                       │
  └──────┬──────┘                     │                       │
         │ h_n[-1]                    │                       │
         ▼                            ▼                       ▼
    batter_hist_emb              seq_embedding          pitch_embedding
         │                            │                       │
         └────────────────────────────┴───────────────────────┘
                                      │
                                   concat
                                      │
                                      ▼
                               ProjectedResBlock
                                      │
                                ResBlock × L
                                      │
                         ┌──────┬─────┴─────┬──────┐
                         ▼      ▼           ▼      ▼
                        SA     SR          BT    Reg
```

### 入力データの詳細

#### (A) 打者履歴エンコーダへの入力

打者の **当該試合より前** の直近 N 打席（デフォルト50打席）の全投球データ。同一 `(batter, game_pk)` の全サンプルは同じ履歴を共有する。データは `batter_game_history.parquet` から事前構築される。

##### 投球レベル特徴量（Inner GRU 入力）— 各打席の各球ごと

| テンソル | 形状 | 内容 |
|----------|------|------|
| `hist_pitch_type` | `(B, N, P)` | 球種 ID → Embedding (打席内系列エンコーダと **共有**) |
| `hist_cont` | `(B, N, P, 15)` | 15 次元の連続値特徴量（正規化済み、下表参照） |
| `hist_swing_attempt` | `(B, N, P)` | スイング試行フラグ (0/1) |
| `hist_swing_result` | `(B, N, P)` | スイング結果 ID → Embedding (打席内系列エンコーダと **共有**) |

`hist_cont` に含まれる 15 特徴量:

| 特徴量 | 説明 | カテゴリ |
|--------|------|---------|
| `release_speed` | 球速 (mph) | 球速 |
| `release_spin_rate` | 回転数 (rpm) | 回転 |
| `pfx_x` | 水平変化量 (ft) | 軌道・変化 |
| `pfx_z` | 垂直変化量 (ft) | 軌道・変化 |
| `vx0` | リリース時 X 方向速度 | 軌道・変化 |
| `vy0` | リリース時 Y 方向速度 | 軌道・変化 |
| `vz0` | リリース時 Z 方向速度 | 軌道・変化 |
| `ax` | X 方向加速度 | 軌道・変化 |
| `ay` | Y 方向加速度 | 軌道・変化 |
| `az` | Z 方向加速度 | 軌道・変化 |
| `plate_x` | プレート通過時の水平位置 (ft) | 通過位置 |
| `plate_z` | プレート通過時の垂直位置 (ft) | 通過位置 |
| `sz_top` | ストライクゾーン上端 (ft) | ゾーン |
| `sz_bot` | ストライクゾーン下端 (ft) | ゾーン |
| `plate_z_norm` | ゾーン正規化済み垂直通過位置 | 通過位置 |

Inner GRU への実際の入力ベクトル（1 投球あたり）:

```
[pitch_type_emb(D_pt), cont(15), swing_attempt(1), swing_result_emb(4)]
 → 合計: D_pt + 15 + 1 + 4 = D_pt + 20 次元
```

##### 打席レベル特徴量（Inner→Outer 間で結合）— 各打席の最終結果

| テンソル | 形状 | 内容 |
|----------|------|------|
| `hist_bb_type` | `(B, N)` | 打球種別 ID (ground_ball / line_drive / fly_ball / popup) → Embedding (dim=4) |
| `hist_launch_speed` | `(B, N)` | 打球速度 (正規化済み、打球なしの場合は 0) |
| `hist_launch_angle` | `(B, N)` | 打球角度 (正規化済み、打球なしの場合は 0) |

これらは **投球単位ではなく打席単位** の情報であり、Inner GRU が投球列を処理した **後** に打席ベクトルへ結合される。

#### (B) 打席内系列エンコーダへの入力

現在の打席における **今の投球より前** の投球列（最大 T=10 球）。`atbat_seq_resdnn` と同一。

| テンソル | 形状 | 内容 |
|----------|------|------|
| `seq_pitch_type` | `(B, T)` | 球種 ID → Embedding |
| `seq_cont` | `(B, T, 15)` | 連続値 15 次元 (上記と同一) |
| `seq_swing_attempt` | `(B, T)` | スイング試行フラグ |
| `seq_swing_result` | `(B, T)` | スイング結果 ID → Embedding |
| `seq_mask` | `(B, T)` | 有効投球マスク |

#### (C) 現在の投球の特徴量

予測対象である現在の 1 球の特徴量。

| 種別 | 特徴量 |
|------|--------|
| カテゴリカル (Embedding) | `pitch_type`, `p_throws`, `batter`, `stand`, `base_out_state`, `count_state` |
| 連続値 | 上記 15 特徴量と同一 |
| 順序値 | `inning_clipped`, `is_inning_top`, `diff_score_clipped`, `pitch_number_clipped` |

### 階層 GRU アーキテクチャの詳細

#### Inner GRU（投球レベル → 打席ベクトル）

各過去打席内の投球系列（最大 P 球）を 1 本の打席ベクトルに圧縮する。B×N 打席分を `(B*N, P, D)` に reshape してバッチ処理。`pitch_type` / `swing_result` の Embedding は打席内系列エンコーダと **重みを共有** する。

```
hist_pitch_type (B*N, P)     ─→ Embedding (shared) ──┐
hist_cont (B*N, P, 15)      ─────────────────────────┤
hist_swing_attempt (B*N, P) ─────────────────────────┼─→ concat ─→ GRU(1層) ─→ h_n[-1]
hist_swing_result (B*N, P)  ─→ Embedding (shared) ───┘                        (B*N, D_inner)
```

- `pack_padded_sequence` で可変長に対応（`hist_pitch_mask` から実長を算出）
- 投球がない打席はゼロベクトル

#### Inner→Outer 間の打席ベクトル構成

Inner GRU 出力に **打席の最終結果** を結合して完全な打席ベクトルを作る。`bb_type` / `launch_speed` / `launch_angle` は投球単位ではなく打席単位の情報であるため、ここで注入する。

```
inner_gru_out (D_inner)  ──┐
bb_type_emb (4)         ───┼─→ concat ─→ atbat_vec (B, N, D_inner + 4 + 2)
launch_speed (1)        ───┤
launch_angle (1)        ───┘
```

**設計意図**: 投球経過（球筋・スイング反応の時系列）と打席最終結果（打球の質）を分離してエンコードすることで、「どんな投球列を経て、どんな結果になったか」を構造的に表現する。

#### Outer GRU（打席ベクトル列 → 打者履歴ベクトル）

N 打席分の打席ベクトル列を時系列として処理し、打者の傾向を固定長ベクトルに圧縮する。

```
atbat_vecs (B, N, D_atbat_vec) ─→ pack_padded_sequence
                                       │
                                       ▼
                                  GRU(num_layers層)
                                       │
                                       ▼
                                  h_n[-1] ─→ batter_hist_emb (B, D_hist_out)
```

- `hist_atbat_mask` で有効な打席のみを処理
- 履歴がない打者はゼロベクトル

### Embedding 共有の構造

| Embedding 層 | 打席内系列エンコーダ (B) | 打者履歴 Inner GRU (A) | 備考 |
|---|---|---|---|
| `seq_pitch_type_embed` | ✅ 使用 | ✅ 使用 | **共有**: 同一の重みで球種を埋め込む |
| `seq_swing_result_embed` | ✅ 使用 | ✅ 使用 | **共有**: 同一の重みでスイング結果を埋め込む |
| `hist_bb_type_embed` | — | ✅ 使用 | **専用**: 打者履歴の打席レベルでのみ使用 |

`pitch_type` と `swing_result` の Embedding を共有することで、パラメータ数を抑えつつ、同一の球種・スイング結果に対して一貫した表現を学習する。

### 設定

```yaml
data:
  batter_history_dir: /workspace/datasets/statcast-customized/batter_history

model:
  architecture: atbat_seq_resdnn_batter_hist
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2

  # 打席内系列エンコーダ
  max_seq_len: 10              # 同一打席内の過去投球の最大系列長
  seq_encoder_type: gru        # "gru" or "transformer"
  seq_hidden_dim: 64           # エンコーダ隠れ層次元
  seq_num_layers: 1            # エンコーダ層数
  seq_bidirectional: false     # GRU のみ: 双方向フラグ

  # 打者履歴エンコーダ
  batter_hist_max_atbats: 50   # 遡る過去打席数 (N)
  batter_hist_max_pitches: 10  # 各打席の最大投球数 (P)
  batter_hist_hidden_dim: 64   # Inner/Outer GRU 隠れ層次元
  batter_hist_num_layers: 1    # Outer GRU 層数 (Inner は常に 1 層)
```

- **設定例**: `configs/seq_resdnn_batter_hist.yaml`
- **必要なデータ**: `batter_history_dir` に `batter_game_history.parquet` と `atbat_row_indices.parquet` が必要（`scripts/add_game_info_and_rebuild.py` で生成）
- **データ分割**: 時系列分割が必須（将来のデータが履歴に混入するリークを防止）
- **データリーク防止**: 打者履歴は「当該試合より前」の打席のみを含む。同一試合内の打席は含まれない

---

## モデルの追加方法

### 1. モデルファイルを作成

`src/models/` に Python ファイルを作成します。

```python
# src/models/my_model.py
import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model


@register_model("my_model")
class MyModel(nn.Module):

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        # cfg.embedding_dims, cfg.backbone_hidden 等を使ってネットワークを構築
        ...

    def forward(
        self,
        cat_dict: dict[str, torch.Tensor],
        cont: torch.Tensor,
        ord_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
        return {
            "swing_attempt": ...,  # (B,) logits
            "swing_result": ...,   # (B, num_swing_result) logits
            "bb_type": ...,        # (B, num_bb_type) logits
            "regression": ...,     # (B, 3) 予測値
        }
```

> 系列モデルの場合は `forward` に `**kwargs` で系列テンソルを受け取り、
> クラス属性 `is_seq_model = True` を定義します。

### 2. `__init__.py` にインポートを追加

`src/models/__init__.py` 末尾のインポートブロックにモジュールを追加します。

```python
from models import atbat_dnn  # noqa: E402, F401
from models import my_model   # noqa: E402, F401  # ← 追加
```

### 3. 設定ファイル (YAML) で指定

```yaml
model:
  architecture: my_model
  # その他のハイパーパラメータ
```

## 規約

| 項目 | 要件 |
|------|------|
| コンストラクタ引数 | `(cfg: ModelConfig, num_cont: int, num_ord: int)` |
| `forward` 引数 | `(cat_dict, cont, ord_feat)` ※系列モデルは `**kwargs` 追加 |
| `forward` 戻り値 | `dict` with keys: `swing_attempt`, `swing_result`, `bb_type`, `regression` |
| 登録名 | `@register_model("名前")` で一意な名前を付ける |
| 系列モデル | クラス属性 `is_seq_model = True` を定義 |
| 打者履歴モデル | クラス属性 `is_batter_hist_model = True` を定義 |
