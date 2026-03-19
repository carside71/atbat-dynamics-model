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

**打者履歴エンコーダ付き系列 ResBlock DNN。** 過去50打席分の Statcast 生投球データを階層 GRU でエンコードし、打者の傾向をモデルに伝える。

```
═══════════════════════════════════════════════════════════════════════
  (A) 打者履歴エンコーダ       (B) 打席内系列エンコーダ      (C) 現在の投球
═══════════════════════════════════════════════════════════════════════

  過去 N 打席 × 最大 P 球      同一打席の過去 T 球        cat / cont / ord
  ─────────────────────      ────────────────        ──────────────

  hist_pitch_type (N,P)      seq_pitch_type (T,)     pitch_type, stand,
  hist_cont (N,P,15)         seq_cont (T,15)         batter, p_throws,
  hist_swing_attempt (N,P)   seq_swing_attempt (T,)  base_out_state, ...
         │                   seq_swing_result (T,)   release_speed, ...
         │                          │                       │
         ▼                          ▼                       ▼
  ┌─────────────┐            ┌─────────────┐         ┌───────────┐
  │  Inner GRU  │            │ GRU /       │         │ Embedding │
  │  (B*N,P,D)  │            │ Transformer │         │  concat   │
  └──────┬──────┘            └──────┬──────┘         └─────┬─────┘
         │ h_n[-1]                  │                      │
         ▼                          │                      │
  ┌─────────────────┐               │                      │
  │ + bb_type_emb   │               │                      │
  │ + launch_speed  │               │                      │
  │ + launch_angle  │               │                      │
  └────────┬────────┘               │                      │
           │ (B, N, D_ab)           │                      │
           ▼                        │                      │
  ┌─────────────┐                   │                      │
  │  Outer GRU  │                   │                      │
  │  (B,N,D_ab) │                   │                      │
  └──────┬──────┘                   │                      │
         │ h_n[-1]                  │                      │
         ▼                          ▼                      ▼
    batter_hist_emb            seq_embedding         pitch_embedding
         │                          │                      │
         └──────────────────────────┴──────────────────────┘
                                    │
                                 concat
                                    │
                                    ▼
                             Input Projection
                                    │
                              ResBlock × L
                                    │
                       ┌──────┬─────┴─────┬──────┐
                       ▼      ▼           ▼      ▼
                      SA     SR          BT    Reg
```

### 階層 GRU アーキテクチャ

#### Inner GRU（投球レベル）

各過去打席内の投球系列（最大 P 球）をエンコード。現在の投球系列エンコーダと `pitch_type` 埋め込みを共有。

```
hist_pitch_type (B*N, P)    ──→ Embedding (shared) ─┐
hist_cont (B*N, P, 15)     ─────────────────────────┼─→ concat ─→ GRU ─→ h_n[-1]
hist_swing_attempt (B*N, P) ────────────────────────┘              (B*N, D_inner)
```

#### 打席単位特徴量の結合

Inner GRU の出力に、打席結果情報（`hist_bb_type` の埋め込み、`hist_launch_speed`、`hist_launch_angle`）を concat。

```
[inner_gru_out, bb_type_emb, launch_speed, launch_angle]  ─→ (B, N, D_atbat_vec)
```

#### Outer GRU（打席レベル）

N 打席分の打席ベクトル系列をエンコード。`hist_atbat_mask` で有効な打席のみを処理。

```
atbat_vecs (B, N, D_atbat_vec) ─→ GRU ─→ h_n[-1] ─→ batter_hist_emb (B, batter_hist_out_dim)
```

### 設定

```yaml
data:
  batter_history_dir: /workspace/datasets/statcast-customized/batter_history

model:
  architecture: atbat_seq_resdnn_batter_hist
  batter_hist_max_atbats: 50      # 過去打席数
  batter_hist_max_pitches: 10     # 各打席の最大投球数
  batter_hist_hidden_dim: 64      # Inner/Outer GRU 隠れ層次元
  batter_hist_num_layers: 1       # GRU 層数
```

- **設定例**: `configs/seq_resdnn_batter_hist.yaml`
- **必要なデータ**: `batter_history_dir` に `batter_game_history.parquet` と `atbat_row_indices.parquet` が必要（`scripts/add_game_info_and_rebuild.py` で生成）
- **データ分割**: 時系列分割が必須（将来のデータが履歴に混入するリークを防止）

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
