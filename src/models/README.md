# src/models/

コンポーネントベースのモデルアーキテクチャ。
YAML 設定ファイルで各コンポーネント（Backbone, Head Strategy, Sequence Encoder 等）を選択・組み合わせてモデルを構築します。

## ディレクトリ構成

```
models/
  __init__.py          # create_model() ファクトリ
  composable.py        # ComposableModel: コンポーネントを組み立てるモデル本体
  components/
    embedding.py       # FeatureEmbedding
    backbones.py       # DNNBackbone, ResDNNBackbone（レジストリ登録）
    heads.py           # build_mlp_head(), MDNHead
    heatmap_head.py    # HeatmapHead, Heatmap2DSubHead, Heatmap1DSubHead
    heatmap_utils.py   # NMS, decode（ヒートマップ後処理）
    pitch_seq_encoders.py    # GRUPitchSeqEncoder, TransformerPitchSeqEncoder（レジストリ登録）
    batter_hist_encoders.py  # GRUBatterHistEncoder, TransformerBatterHistEncoder（レジストリ登録）
    head_strategies.py # IndependentHeadStrategy, CascadeHeadStrategy
```

レジストリパターンは `utils/registry.py` の `make_registry()` で統一的に管理されています。

---

## 全体構造

すべてのモデルは `ComposableModel` が以下のコンポーネントを YAML に基づいて組み立てます。

<div align="center">
<table style="border-collapse: separate; border-spacing: 8px 4px;">
<tr>
  <td style="background:#eceff1; border:2px solid #78909c; border-radius:8px; padding:6px 12px; text-align:center;">カテゴリカル特徴量</td>
  <td rowspan="3" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px;">→</td>
  <td rowspan="3" style="background:#e3f2fd; border:2px solid #42a5f5; border-radius:8px; padding:8px 14px; text-align:center;"><b>Embedding</b></td>
  <td rowspan="5" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0 6px;">→ <i>concat</i> →</td>
  <td rowspan="5" style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center;"><b>Backbone</b><br><sub>DNN / ResDNN</sub></td>
  <td rowspan="5" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px;">→</td>
  <td rowspan="5" style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:10px 16px; text-align:center;"><b>HeadStrategy</b><br><sub>Independent / Cascade</sub></td>
</tr>
<tr><td style="background:#eceff1; border:2px solid #78909c; border-radius:8px; padding:6px 12px; text-align:center;">連続値特徴量</td></tr>
<tr><td style="background:#eceff1; border:2px solid #78909c; border-radius:8px; padding:6px 12px; text-align:center;">順序特徴量</td></tr>
<tr>
  <td style="background:#eceff1; border:2px solid #78909c; border-radius:8px; padding:6px 12px; text-align:center;">過去投球系列 <i>(opt)</i></td>
  <td style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px;">→</td>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:10px 16px; text-align:center;"><b>PitchSeqEncoder</b><br><sub>GRU / Transformer</sub></td>
</tr>
<tr>
  <td style="background:#eceff1; border:2px solid #78909c; border-radius:8px; padding:6px 12px; text-align:center;">打者履歴 <i>(opt)</i></td>
  <td style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px;">→</td>
  <td style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:10px 16px; text-align:center;"><b>BatterHistEncoder</b><br><sub>GRU / Transformer</sub></td>
</tr>
</table>
</div>

### 出力

出力辞書に含まれるキーは `model_scope` 設定によって変化します。

| `model_scope` | 出力キー | 用途 |
|---|---|---|
| `all`（デフォルト） | `swing_attempt`, `swing_result`, `bb_type`, `regression` | 全タスク統合モデル |
| `swing_attempt` | `swing_attempt` | スイング判定専用モデル |
| `outcome` | `swing_result`, `bb_type`, `regression` | スイング後の結果予測専用モデル |
| `classification` | `swing_attempt`, `swing_result`, `bb_type` | 3分類タスクのみ（回帰なし） |
| `regression` | `regression` | 回帰予測専用モデル |

| キー | 形状 | 内容 |
|---|---|---|
| `swing_attempt` | `(B,)` | スイング試行 logit (binary) |
| `swing_result` | `(B, num_swing_result)` | スイング結果 logits |
| `bb_type` | `(B, num_bb_type)` | 打球タイプ logits |
| `regression` | `(B, D)` or `dict` | launch_speed, launch_angle, hit_distance_sc, spray_angle。MLP: `(B, D)` テンソル、MDN: `pi`, `mu`, `sigma` の dict、Heatmap（レガシー）: `heatmap_2d`, `offset_2d`, `heatmap_launch_speed`, `offset_launch_speed`, `heatmap_hit_distance`, `offset_hit_distance` の dict、Heatmap（設定モード）: `heatmap_{key}`, `offset_{key}` の dict（key は `make_heatmap_key()` で生成、例: `heatmap_2d_launch_angle__spray_angle`, `heatmap_1d_launch_speed`）。D = `num_reg_targets`（デフォルト 4） |

---

## コンポーネント詳細

### 1. FeatureEmbedding (`components/embedding.py`)

カテゴリカル特徴量を Embedding し、連続値・序数特徴量と concat する。全モデル構成で共通。

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 3px;">
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">p_throws</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding(3, 2)</td>
  <td rowspan="8" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0 6px;">→ <i>concat</i> →</td>
  <td rowspan="8" style="background:#fffde7; border:2px solid #fdd835; border-radius:8px; padding:8px 14px; text-align:center; font-size:13px;"><b>(B, embed_dim<br>+ num_cont<br>+ num_ord)</b></td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">pitch_type</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding(19, 8)</td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">batter</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding(784, 16)</td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">stand</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding(3, 2)</td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">base_out_state</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding(25, 8)</td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">count_state</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding(13, 4)</td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">cont (15)</td>
  <td colspan="2" style="border:none;"></td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">ord (4)</td>
  <td colspan="2" style="border:none;"></td>
</tr>
</table>
</div>

- Embedding の `padding_idx` = `num_classes`（不正値の吸収用）
- 不正値（-1 や range 外）は自動的に `padding_idx` にマッピング

---

### 2. Backbone (`components/backbones.py`)

YAML の `backbone_type` で選択。

#### `dnn` — DNN Backbone

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 0;">
<tr>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center;"><b>Linear</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center;"><b>BatchNorm1d</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center;"><b>ReLU</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center;"><b>Dropout</b></td>
  <td style="border:none; color:#546e7a; padding-left:8px;">× len(backbone_hidden)</td>
</tr>
</table>
</div>

- シンプルな全結合レイヤーのスタック
- `backbone_hidden: [512, 256, 128]` で 3 層

#### `resdnn` — ResBlock Backbone

<div align="center">
<table style="border:2px solid #66bb6a; border-radius:12px; border-collapse:collapse; background:#fafafa;">
<tr>
  <td colspan="3" style="background:#e8f5e9; border-bottom:1px solid #c8e6c9; padding:6px 16px; text-align:center; border-radius:10px 10px 0 0;">
    &laquo;block&raquo; <b>ResBlock / ProjectedResBlock</b>
  </td>
</tr>
<tr>
  <td style="padding:12px 16px; text-align:center; border:none; vertical-align:top; width:45%;">
    <b>Main path</b><br>
    <span style="background:#e8f5e9; border:1px solid #a5d6a7; border-radius:4px; padding:2px 8px; display:inline-block; margin:4px 0; font-size:13px;">Linear → LN → GELU → Dropout</span><br>↓<br>
    <span style="background:#e8f5e9; border:1px solid #a5d6a7; border-radius:4px; padding:2px 8px; display:inline-block; margin:4px 0; font-size:13px;">Linear → LN</span>
  </td>
  <td style="border-left:1px dashed #c8e6c9; border-right:1px dashed #c8e6c9; padding:12px 8px; text-align:center; vertical-align:middle; font-size:24px; font-weight:bold; color:#66bb6a;">
    ＋
  </td>
  <td style="padding:12px 16px; text-align:center; border:none; vertical-align:top; width:45%;">
    <b>Skip path</b><br>
    <span style="background:#e8f5e9; border:1px solid #a5d6a7; border-radius:4px; padding:2px 8px; display:inline-block; margin:4px 0; font-size:13px;">Identity</span><br>
    <sub>(ProjectedResBlock の場合:<br>Linear → LN)</sub>
  </td>
</tr>
<tr>
  <td colspan="3" style="background:#e8f5e9; border-top:1px solid #c8e6c9; padding:6px 16px; text-align:center; border-radius:0 0 10px 10px;">
    → <b>GELU</b> → output
  </td>
</tr>
</table>
</div>

- **ResBlock**: 入力と出力の次元が同一の場合
- **ProjectedResBlock**: 次元が異なる場合。skip 接続に射影 `Linear` を挿入
- `backbone_hidden: [512, 512, 256, 256, 128]` で 5 ブロック

---

### 3. Head Strategy (`components/head_strategies.py`)

YAML の `head_strategy` で選択。`model_scope` に応じて構築されるヘッドが変化します。

| `model_scope` | `independent` | `cascade` |
|---|---|---|
| `all` | 4ヘッド並列（従来動作） | SA→SR→BT→Reg カスケード（従来動作） |
| `swing_attempt` | SA ヘッドのみ | SA ヘッドのみ |
| `outcome` | SR + BT + Reg ヘッド並列 | SR→BT→Reg カスケード（SA なし） |
| `classification` | SA + SR + BT ヘッド並列 | SA→SR→BT カスケード（Reg なし） |
| `regression` | Reg ヘッドのみ | Reg ヘッドのみ（backbone 出力直結） |

#### `independent` — 独立ヘッド（デフォルト）

<div align="center">
<table style="border-collapse: separate; border-spacing: 8px 6px;">
<tr>
  <td rowspan="4" style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center; vertical-align:middle;"><b>backbone<br>output h</b></td>
  <td rowspan="4" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px; vertical-align:middle;">→</td>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>MLP</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">swing_attempt <code>(B,)</code></td>
</tr>
<tr>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>MLP</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">swing_result <code>(B, 9)</code></td>
</tr>
<tr>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>MLP</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">bb_type <code>(B, 4)</code></td>
</tr>
<tr>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>MLP / MDN / Heatmap</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">regression <code>(B, D)</code></td>
</tr>
</table>
</div>

全ヘッドが backbone 出力を独立に受け取る。

#### `cascade` — カスケードヘッド

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 2px;">
<tr>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center;"><b>h</b> <sub>(backbone output)</sub></td>
  <td style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px;">→</td>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>SA Head</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">swing_attempt</td>
</tr>
<tr><td colspan="5" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0;">↓ concat [h, sa_logit]</td></tr>
<tr>
  <td colspan="2" style="border:none;"></td>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>SR Head</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">swing_result</td>
</tr>
<tr><td colspan="5" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0;">↓ concat [h, sr_logit]</td></tr>
<tr>
  <td colspan="2" style="border:none;"></td>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>BT Head</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">bb_type</td>
</tr>
<tr><td colspan="5" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0;">↓ concat [h, bt_logit]</td></tr>
<tr>
  <td colspan="2" style="border:none;"></td>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>Reg Head</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">regression <code>(B, D)</code></td>
</tr>
</table>
</div>

- **カスケードの流れ**: `h → SA → [h,sa] → SR → [h,sr] → BT → [h,bt] → Reg`
- **`model_scope="outcome"` 時**: SA ヘッドが省略され `h → SR → [h,sr] → BT → [h,bt] → Reg` となる
- **`model_scope="classification"` 時**: Reg ヘッドが省略され `h → SA → [h,sa] → SR → [h,sr] → BT` となる
- **`model_scope="regression"` 時**: 分類ヘッドが省略され `h → Reg` となる（backbone 出力直結）
- **`detach_cascade`**: `true` にすると上流ヘッドからの勾配を `detach` し、下流ヘッド学習時に上流を更新しない
- **設計意図**: スイング試行 → スイング結果 → 打球タイプ → 回帰値 という因果構造を反映

---

### 4. Regression Head Type

YAML の `regression_head_type` で選択。

#### `mlp`（デフォルト）

通常の MLP ヘッド。出力 `(B, D)`。D = `num_reg_targets`（デフォルト 4）。

#### `mdn` — Mixture Density Network

<div align="center">
<table style="border-collapse: separate; border-spacing: 6px 4px;">
<tr>
  <td rowspan="3" style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:8px 14px; text-align:center; vertical-align:middle;"><b>backbone_out</b></td>
  <td rowspan="3" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px; vertical-align:middle;">→</td>
  <td rowspan="3" style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:10px 16px; text-align:center; vertical-align:middle;"><b>Shared MLP</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fff3e0; border:1px solid #ffe0b2; border-radius:6px; padding:4px 10px; font-size:13px; text-align:center;">fc_pi → <b>Softmax</b></td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;"><b>π</b> (B, K) 混合係数</td>
</tr>
<tr>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fff3e0; border:1px solid #ffe0b2; border-radius:6px; padding:4px 10px; font-size:13px; text-align:center;">fc_mu → <b>reshape</b></td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;"><b>μ</b> (B, K, D) 平均</td>
</tr>
<tr>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fff3e0; border:1px solid #ffe0b2; border-radius:6px; padding:4px 10px; font-size:13px; text-align:center;">fc_sigma → <b>Softplus</b></td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;"><b>σ</b> (B, K, D) 標準偏差</td>
</tr>
</table>
</div>

- K 個のガウス分布の混合で回帰ターゲットをモデル化
- `mdn_num_components` で K を指定
- 推論時: `E[y] = Σ_k π_k * μ_k` を点推定として採用

#### `heatmap` — Heatmap Head

CenterNet スタイルのヒートマップベース回帰。

**動作モード:**

- **レガシーモード**（`heatmap_heads` 未指定）: 従来のハードコード構成。launch_angle × spray_angle の 2D ヒートマップ + launch_speed / hit_distance_sc の 1D ヒートマップの 3 サブヘッド固定。
- **設定モード**（`heatmap_heads` 指定）: YAML で 1D/2D サブヘッドの割り当てを自由に設定可能。各ターゲットを個別の 1D ヘッドで予測したり、任意の 2 ターゲットを 2D ヘッドでジョイント予測したりできる。

**HeatmapHead 全体構造（レガシーモード）:**

<div align="center">
<table style="border-collapse: separate; border-spacing: 6px 6px;">
<tr>
  <td rowspan="3" style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:10px 16px; text-align:center; vertical-align:middle;"><b>backbone<br>output</b><br><sub>(B, D)</sub></td>
  <td rowspan="3" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px; vertical-align:middle;">→</td>
  <td style="background:#e1f5fe; border:2px solid #29b6f6; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap2DSubHead</b><br><sub>launch_angle × spray_angle</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,H,W)</code><br>offset <code>(B,2,H,W)</code></td>
</tr>
<tr>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap1DSubHead</b><br><sub>launch_speed</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,L)</code><br>offset <code>(B,1,L)</code></td>
</tr>
<tr>
  <td style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap1DSubHead</b><br><sub>hit_distance_sc</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,L)</code><br>offset <code>(B,1,L)</code></td>
</tr>
</table>
</div>

**HeatmapHead 全体構造（設定モード — 全ターゲット 1D の例）:**

<div align="center">
<table style="border-collapse: separate; border-spacing: 6px 6px;">
<tr>
  <td rowspan="4" style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:10px 16px; text-align:center; vertical-align:middle;"><b>backbone<br>output</b><br><sub>(B, D)</sub></td>
  <td rowspan="4" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px; vertical-align:middle;">→</td>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap1DSubHead</b><br><sub>launch_speed</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,L)</code><br>offset <code>(B,1,L)</code></td>
</tr>
<tr>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap1DSubHead</b><br><sub>launch_angle</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,L)</code><br>offset <code>(B,1,L)</code></td>
</tr>
<tr>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap1DSubHead</b><br><sub>hit_distance_sc</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,L)</code><br>offset <code>(B,1,L)</code></td>
</tr>
<tr>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:8px 14px; text-align:center;"><b>Heatmap1DSubHead</b><br><sub>spray_angle</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:6px 12px; font-size:13px;">heatmap <code>(B,1,L)</code><br>offset <code>(B,1,L)</code></td>
</tr>
</table>
</div>

設定モードの出力 dict キーは `make_heatmap_key()` で自動生成される:
- 2D ヘッド: `heatmap_2d_{target_a}__{target_b}`, `offset_2d_{target_a}__{target_b}`
- 1D ヘッド: `heatmap_1d_{target}`, `offset_1d_{target}`

**2D サブヘッド内部構造 (Heatmap2DSubHead):**

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 4px;">
<tr>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>backbone_out</b><br><sub>(B, D)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>Linear</b><br><sub>→ (B, C·4·4)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>reshape</b><br><sub>→ (B, C, 4, 4)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e1f5fe; border:2px solid #29b6f6; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>ConvTranspose2d</b><br>× 4 stages<br><sub>BN + ReLU</sub><br><sub>4→8→16→32→64</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="border:none; vertical-align:middle;">
    <table style="border-collapse: separate; border-spacing: 4px 4px;">
    <tr>
      <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>Conv2d</b> → <b>sigmoid</b></td>
      <td style="border:none; color:#546e7a;">→</td>
      <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:4px 8px; font-size:12px;"><b>heatmap</b><br><sub>(B, 1, H, W)</sub></td>
    </tr>
    <tr>
      <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>Conv2d</b></td>
      <td style="border:none; color:#546e7a;">→</td>
      <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:4px 8px; font-size:12px;"><b>offset</b> (dy, dx)<br><sub>(B, 2, H, W)</sub></td>
    </tr>
    </table>
  </td>
</tr>
</table>
</div>

**1D サブヘッド内部構造 (Heatmap1DSubHead):**

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 4px;">
<tr>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>backbone_out</b><br><sub>(B, D)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>Linear</b><br><sub>→ (B, C·4)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>reshape</b><br><sub>→ (B, C, 4)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>ConvTranspose1d</b><br>× 4 stages<br><sub>BN + ReLU</sub><br><sub>4→8→16→32→64</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="border:none; vertical-align:middle;">
    <table style="border-collapse: separate; border-spacing: 4px 4px;">
    <tr>
      <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>Conv1d</b> → <b>sigmoid</b></td>
      <td style="border:none; color:#546e7a;">→</td>
      <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:4px 8px; font-size:12px;"><b>heatmap</b><br><sub>(B, 1, L)</sub></td>
    </tr>
    <tr>
      <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:6px; padding:6px 10px; text-align:center; font-size:13px;"><b>Conv1d</b></td>
      <td style="border:none; color:#546e7a;">→</td>
      <td style="background:#fffde7; border:1px solid #fdd835; border-radius:6px; padding:4px 8px; font-size:12px;"><b>offset</b><br><sub>(B, 1, L)</sub></td>
    </tr>
    </table>
  </td>
</tr>
</table>
</div>

**学習時のGT生成と推論時のデコード:**

<div align="center">
<table style="border-collapse: separate; border-spacing: 6px 4px;">
<tr>
  <td colspan="5" style="border:none; text-align:center; color:#546e7a; font-size:14px; padding:4px 0;"><b>学習時（GT → ガウスヒートマップ）</b></td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:6px 12px; text-align:center; font-size:13px;">GT 値<br><sub>(正規化済み)</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:6px 12px; text-align:center; font-size:13px;">ピクセル座標<br>に変換</td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>2D ガウス</b><br><sub>σ=2.0 (pixel)</sub><br>+ <b>Offset</b></td>
</tr>
<tr><td colspan="5" style="border:none; padding:6px;"></td></tr>
<tr>
  <td colspan="5" style="border:none; text-align:center; color:#546e7a; font-size:14px; padding:4px 0;"><b>推論時（ヒートマップ → 予測値）</b></td>
</tr>
<tr>
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>予測<br>heatmap</b></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>NMS</b><br><sub>max_pool<br>ピーク検出</sub></td>
  <td style="border:none; text-align:center; font-size:18px; color:#546e7a;">→</td>
  <td style="background:#e8f5e9; border:2px solid #66bb6a; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px;"><b>argmax</b><br><sub>ピーク座標</sub><br>+ <b>offset</b></td>
</tr>
</table>
</div>

- **ヒートマップブランチ**: Deconvolution で backbone 出力を空間マップに展開し、sigmoid でピーク位置を予測
- **オフセットブランチ**: ピーク位置からの精密なオフセット値を予測
- **GT 生成**: GT 値から正規化座標空間上のガウス分布ヒートマップを生成（`losses/heatmap.py`）
- **NMS**: max_pool で局所最大値を検出するルールベースの後処理（`heatmap_utils.py`）
- **推論時**: NMS → argmax でピーク座標取得 → ピクセル中心値 + オフセットで最終予測値を復元

**設定パラメータ:**

| フィールド | デフォルト | 説明 |
|---|---|---|
| `heatmap_grid_h` | `64` | 2D ヒートマップの高さ (launch_angle 軸) |
| `heatmap_grid_w` | `64` | 2D ヒートマップの幅 (spray_angle 軸) |
| `heatmap_num_bins` | `64` | 1D ヒートマップのビン数 |
| `heatmap_range_launch_speed` | `[20.0, 120.0]` | launch_speed の物理値域 (mph) |
| `heatmap_range_launch_angle` | `[-90.0, 90.0]` | launch_angle の物理値域 (deg) |
| `heatmap_range_hit_distance` | `[0.0, 500.0]` | hit_distance_sc の物理値域 (ft) |
| `heatmap_range_spray_angle` | `[-180.0, 180.0]` | spray_angle の物理値域 (deg) |
| `heatmap_sigma` | `2.0` | GT ガウスの sigma（ピクセル単位） |
| `heatmap_intermediate_dim` | `256` | Deconv 前の中間チャネル数 |
| `heatmap_heads` | `null` | サブヘッド構成のリスト。未指定でレガシーモード。詳細は下記参照 |

> **`heatmap_sigma` の詳細:**
> GT（正解）の連続値をヒートマップのグリッド座標に変換した後、中心ピクセルを頂点とするガウス分布を教師信号として生成する。
> `heatmap_sigma` はこのガウスの標準偏差を **ピクセル（ビン）単位** で指定する（`heatmap.py` L73: `exp(-(dy²+dx²) / (2σ²))`）。
> 値を小さくするとよりシャープ（ピンポイント）、大きくするとより滑らかな教師信号になる。
>
> デフォルト `sigma=2.0` での物理量換算の目安（±1σ）:
>
> | ターゲット | 値域 | ビン数 | 1ビンあたり | ±1σ の幅 |
> |---|---|---|---|---|
> | launch_angle | -90°〜90° (180°) | 64 | ≈2.81° | ±5.63° |
> | spray_angle | -180°〜180° (360°) | 64 | ≈5.63° | ±11.25° |
> | launch_speed | 20〜120 mph (100) | 64 | ≈1.56 mph | ±3.13 mph |
> | hit_distance | 0〜500 ft (500) | 64 | ≈7.81 ft | ±15.63 ft |
>
> ※ 実際には正規化後の値域 `heatmap_norm_range_*` が使われるため、上記は `heatmap_range_*` からの概算。

**損失関数（`TrainConfig`）:**

| フィールド | デフォルト | 説明 |
|---|---|---|
| `heatmap_loss_weight_offset` | `1.0` | オフセット L1 損失の重み |
| `heatmap_focal_alpha` | `2.0` | Focal loss の alpha |
| `heatmap_focal_beta` | `4.0` | Focal loss の beta |

---

### 5. PitchSeqEncoder (`components/pitch_seq_encoders.py`)

`pitch_seq_max_len > 0` のとき有効化。YAML の `pitch_seq_encoder_type` で選択。

同一打席内の過去投球系列をエンコードする。

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 3px;">
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">seq_pitch_type</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding</td>
  <td rowspan="4" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0 6px;">→ <i>concat</i> →</td>
  <td rowspan="4" style="background:#fafafa; border:2px dashed #9e9e9e; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px; vertical-align:middle;"><b>(B, T, D_seq)</b></td>
  <td rowspan="4" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px; vertical-align:middle;">→</td>
  <td rowspan="4" style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:10px 16px; text-align:center; vertical-align:middle;"><b>Encoder</b></td>
  <td rowspan="4" style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px; vertical-align:middle;">→</td>
  <td rowspan="4" style="background:#fffde7; border:2px solid #fdd835; border-radius:8px; padding:8px 14px; text-align:center; font-size:13px; vertical-align:middle;"><b>(B, pitch_seq_out_dim)</b></td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">seq_swing_result</td>
  <td style="border:none; color:#546e7a;">→</td>
  <td style="background:#e3f2fd; border:1px solid #42a5f5; border-radius:6px; padding:3px 8px; font-size:13px;">Embedding</td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">seq_cont</td>
  <td colspan="2" style="border:none;"></td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px;">seq_swing_attempt</td>
  <td colspan="2" style="border:none;"></td>
</tr>
</table>
</div>

#### `gru` — GRU エンコーダ

- `pack_padded_sequence` で可変長系列に対応
- 双方向時は最終隠れ状態の forward/backward を concat
- 過去投球がない場合はゼロベクトル

#### `transformer` — Transformer エンコーダ

- `Linear` で入力を `pitch_seq_hidden_dim` に射影後、`TransformerEncoder` で処理
- `src_key_padding_mask` でパディング位置をマスク
- マスク付き平均プーリングで固定長ベクトル化

---

### 6. BatterHistEncoder (`components/batter_hist_encoders.py`)

`batter_hist_max_atbats > 0` のとき有効化。`pitch_seq_max_len > 0`（PitchSeqEncoder 有効）が前提。YAML の `batter_hist_encoder_type` で選択。

打者の直近 N 打席の Statcast 全投球データをエンコードする。

<div align="center">
<table style="border-collapse: separate; border-spacing: 4px 3px;">
<tr>
  <td colspan="3" style="border:none; text-align:center; color:#546e7a; padding:4px 0;"><b>過去 N 打席 × 各最大 P 球</b></td>
</tr>
<tr>
  <td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px; text-align:left;">hist_pitch_type (B,N,P) → <i>Emb 共有</i></td>
  <td rowspan="4" style="border:none; text-align:center; color:#546e7a; font-size:13px; padding:0 6px;">→<br><i>concat</i><br>→</td>
  <td rowspan="4" style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:10px 16px; text-align:center; vertical-align:middle;"><b>Inner GRU</b><br><sub>(B*N, P, D) → h_n[-1]<br>→ (B*N, D_inner)</sub></td>
</tr>
<tr><td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px; text-align:left;">hist_cont (B,N,P,15)</td></tr>
<tr><td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px; text-align:left;">hist_swing_attempt (B,N,P)</td></tr>
<tr><td style="background:#eceff1; border:1px solid #90a4ae; border-radius:6px; padding:3px 10px; text-align:right; font-size:13px; text-align:left;">hist_swing_result (B,N,P) → <i>Emb 共有</i></td></tr>
<tr>
  <td colspan="3" style="border:none; text-align:center; color:#546e7a; font-size:14px; padding:4px 0;">↓ reshape → (B, N, D_inner)</td>
</tr>
<tr>
  <td colspan="3" style="border:none; text-align:center; padding:2px 0;">
    <span style="background:#fafafa; border:2px dashed #9e9e9e; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px; display:inline-block; padding:6px 14px;"><b>concat</b>: + bb_type_emb (4) + launch_speed (1) + launch_angle (1) + spray_angle (1)</span>
  </td>
</tr>
<tr>
  <td colspan="3" style="border:none; text-align:center; color:#546e7a; font-size:14px; padding:4px 0;">↓ (B, N, D_inner + 7)</td>
</tr>
<tr>
  <td colspan="3" style="border:none; text-align:center; padding:2px 0;">
    <span style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:10px 16px; text-align:center; display:inline-block; padding:8px 20px;"><b>Outer Encoder</b> (GRU / Transformer) → <b>(B, D_hist_out)</b></span>
  </td>
</tr>
</table>
</div>

- **Inner GRU**: 各打席の投球列を 1 本の打席ベクトルに圧縮（全サブクラス共通）
- **Outer Encoder**: N 打席分の打席ベクトル列を処理して打者の傾向ベクトルに圧縮
  - `gru`: Outer GRU で最終隠れ状態を出力
  - `transformer`: Linear 射影 → TransformerEncoder → マスク付き平均プーリング
- `pitch_type` / `swing_result` の Embedding は PitchSeqEncoder と **重みを共有**
- 投球がない打席・履歴がない打者はゼロベクトル

---

## YAML 設定

### モデル設定フィールド一覧

| フィールド | デフォルト | 選択肢 | 説明 |
|---|---|---|---|
| `model_scope` | `"all"` | `"all"`, `"swing_attempt"`, `"outcome"`, `"classification"`, `"regression"` | 予測タスクの範囲 |
| `backbone_type` | `"resdnn"` | `"dnn"`, `"resdnn"` | Backbone の種類 |
| `backbone_hidden` | `[512, 256, 128]` | — | 各層の隠れ次元 |
| `dropout` | `0.2` | — | Dropout 率 |
| `head_strategy` | `"independent"` | `"independent"`, `"cascade"` | ヘッド接続戦略 |
| `head_hidden` | `[64]` | — | ヘッド MLP の隠れ次元 |
| `head_activation` | `"gelu"` | `"relu"`, `"gelu"` | ヘッド MLP の活性化関数 |
| `detach_cascade` | `true` | — | cascade 時に上流勾配を detach |
| `regression_head_type` | `"mlp"` | `"mlp"`, `"mdn"`, `"heatmap"` | 回帰ヘッドの種類 |
| `mdn_num_components` | `5` | — | MDN のガウス成分数 |
| `heatmap_grid_h` | `64` | — | 2D ヒートマップの高さ |
| `heatmap_grid_w` | `64` | — | 2D ヒートマップの幅 |
| `heatmap_num_bins` | `64` | — | 1D ヒートマップのビン数 |
| `heatmap_range_launch_speed` | `[40.0, 120.0]` | — | launch_speed の物理値域 (mph) |
| `heatmap_range_launch_angle` | `[-90.0, 90.0]` | — | launch_angle の物理値域 (deg) |
| `heatmap_range_hit_distance` | `[0.0, 500.0]` | — | hit_distance_sc の物理値域 (ft) |
| `heatmap_range_spray_angle` | `[-45.0, 45.0]` | — | spray_angle の物理値域 (deg) |
| `heatmap_sigma` | `2.0` | — | GT ガウスの sigma |
| `heatmap_intermediate_dim` | `256` | — | Deconv 前の中間チャネル数 |
| `heatmap_heads` | `null` | — | サブヘッド構成リスト。`null` でレガシーモード |
| `pitch_seq_max_len` | `0` | — | 0: 無効、>0: 投球シーケンスエンコーダ有効 |
| `pitch_seq_encoder_type` | `"gru"` | `"gru"`, `"transformer"` | 投球シーケンスエンコーダの種類 |
| `pitch_seq_hidden_dim` | `64` | — | 投球シーケンスエンコーダの隠れ次元 |
| `pitch_seq_num_layers` | `1` | — | 投球シーケンスエンコーダの層数 |
| `pitch_seq_bidirectional` | `false` | — | GRU のみ: 双方向フラグ |
| `batter_hist_max_atbats` | `0` | — | 0: 無効、>0: 打者履歴エンコーダ有効 |
| `batter_hist_max_pitches` | `10` | — | 各打席の最大投球数 |
| `batter_hist_encoder_type` | `"gru"` | `"gru"`, `"transformer"` | 打者履歴エンコーダの種類 |
| `batter_hist_hidden_dim` | `64` | — | 打者履歴エンコーダの隠れ次元 |
| `batter_hist_num_layers` | `1` | — | Outer エンコーダの層数 |

### 設定例

#### シンプルな DNN（ベースライン）

```yaml
model:
  backbone_type: dnn
  backbone_hidden: [512, 256, 128]
  head_hidden: [64]
  head_activation: relu
  dropout: 0.2
```

#### DNN + MDN 回帰ヘッド

```yaml
model:
  backbone_type: dnn
  backbone_hidden: [512, 256, 128]
  head_hidden: [64]
  head_activation: relu
  dropout: 0.2
  regression_head_type: mdn
  mdn_num_components: 5
```

#### DNN + Heatmap 回帰ヘッド（レガシーモード）

```yaml
model:
  backbone_type: dnn
  backbone_hidden: [512, 256, 128]
  dropout: 0.2
  regression_head_type: heatmap
  heatmap_grid_h: 64
  heatmap_grid_w: 64
  heatmap_num_bins: 64
  heatmap_range_launch_speed: [40.0, 120.0]
  heatmap_range_launch_angle: [-90.0, 90.0]
  heatmap_range_hit_distance: [0.0, 500.0]
  heatmap_range_spray_angle: [-45.0, 45.0]
  heatmap_sigma: 2.0

train:
  heatmap_loss_weight_offset: 1.0
  heatmap_focal_alpha: 2.0
  heatmap_focal_beta: 4.0
```

#### DNN + Heatmap 回帰ヘッド（設定モード — 2D + 1D 混合）

`heatmap_heads` で各ターゲットの 1D/2D 割り当てを明示指定する。
レガシーモードと同じ構成を設定モードで再現する例:

```yaml
model:
  backbone_type: dnn
  backbone_hidden: [512, 256, 128]
  dropout: 0.2
  regression_head_type: heatmap
  heatmap_grid_h: 64
  heatmap_grid_w: 64
  heatmap_num_bins: 64
  heatmap_range_launch_speed: [40.0, 120.0]
  heatmap_range_launch_angle: [-90.0, 90.0]
  heatmap_range_hit_distance: [0.0, 500.0]
  heatmap_range_spray_angle: [-45.0, 45.0]
  heatmap_sigma: 2.0
  heatmap_heads:
    - type: "2d"
      targets: [launch_angle, spray_angle]
    - type: "1d"
      targets: [launch_speed]
    - type: "1d"
      targets: [hit_distance_sc]

train:
  heatmap_loss_weight_offset: 1.0
  heatmap_focal_alpha: 2.0
  heatmap_focal_beta: 4.0
```

#### DNN + Heatmap 回帰ヘッド（設定モード — 全ターゲット 1D）

全ターゲットを独立した 1D ヒートマップで予測する構成:

```yaml
model:
  backbone_type: dnn
  backbone_hidden: [512, 256, 128]
  dropout: 0.2
  regression_head_type: heatmap
  heatmap_num_bins: 64
  heatmap_range_launch_speed: [40.0, 120.0]
  heatmap_range_launch_angle: [-90.0, 90.0]
  heatmap_range_hit_distance: [0.0, 500.0]
  heatmap_range_spray_angle: [-45.0, 45.0]
  heatmap_sigma: 2.0
  heatmap_heads:
    - type: "1d"
      targets: [launch_speed]
    - type: "1d"
      targets: [launch_angle]
    - type: "1d"
      targets: [hit_distance_sc]
    - type: "1d"
      targets: [spray_angle]

train:
  heatmap_loss_weight_offset: 1.0
  heatmap_focal_alpha: 2.0
  heatmap_focal_beta: 4.0
```

#### ResBlock DNN

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
```

#### ResBlock DNN + カスケードヘッド

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  head_strategy: cascade
  detach_cascade: true
```

#### ResBlock DNN + GRU 投球シーケンスエンコーダ

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  pitch_seq_max_len: 10
  pitch_seq_encoder_type: gru
  pitch_seq_hidden_dim: 64
  pitch_seq_num_layers: 1
  pitch_seq_bidirectional: false
```

#### ResBlock DNN + GRU 投球シーケンス + 打者履歴

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  pitch_seq_max_len: 10
  pitch_seq_encoder_type: gru
  pitch_seq_hidden_dim: 64
  pitch_seq_num_layers: 1
  pitch_seq_bidirectional: false
  batter_hist_max_atbats: 50
  batter_hist_max_pitches: 10
  batter_hist_encoder_type: gru
  batter_hist_hidden_dim: 64
  batter_hist_num_layers: 1
```

#### swing_attempt 専用モデル

```yaml
model:
  model_scope: swing_attempt
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
```

#### outcome 専用モデル（swing_attempt=1 サンプルのみで学習）

```yaml
model:
  model_scope: outcome
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
```

#### classification 専用モデル（SA/SR/BT の3分類、回帰なし）

```yaml
model:
  model_scope: classification
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
```

#### regression 専用モデル（swing_attempt=1 サンプルのみで学習）

```yaml
model:
  model_scope: regression
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
```

#### ResBlock + カスケード + 物理的整合性損失

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  head_strategy: cascade
  detach_cascade: true

train:
  loss_weight_physics: 0.01       # 物理的整合性損失の重み（0.0 で無効）
  physics_margin_degrees: 2.0     # 境界マージン（度）
```

bb_type と launch_angle、swing_result と spray_angle の間の物理的整合性をソフトペナルティで強制する。分類予測の softmax 確率で重み付けした `torch.relu` ベースの微分可能ペナルティを既存のタスク損失に加算する。MDN ヘッドにも対応（期待値 E[y] = Σ π_k * μ_k を使用）。

#### 新しい組み合わせ例: ResBlock + カスケード + MDN

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  head_strategy: cascade
  regression_head_type: mdn
  mdn_num_components: 5
```

---

## コンポーネントの追加方法

Backbone と Sequence Encoder は `utils/registry.py` の `make_registry()` で生成されたレジストリとデコレータで管理されています。`@register_xxx("name")` デコレータを付けてクラスを定義するだけで、YAML から `name` で参照可能になります。

### 新しい Backbone を追加

`components/backbones.py` にクラスを追加し、レジストリに登録する。

```python
@register_backbone("my_backbone")
class MyBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        ...
        self._output_dim = ...

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

YAML で `backbone_type: my_backbone` を指定するだけで利用可能。

### 新しい PitchSeqEncoder を追加

`components/pitch_seq_encoders.py` にクラスを追加し、レジストリに登録する。

```python
@register_pitch_seq_encoder("my_encoder")
class MyPitchSeqEncoder(BasePitchSeqEncoder):
    def __init__(self, cfg: ModelConfig, num_cont: int):
        super().__init__(cfg, num_cont)
        ...
        self._output_dim = ...

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, seq_pitch_type, seq_cont, seq_swing_attempt,
                seq_swing_result, seq_mask) -> torch.Tensor:
        ...
```

YAML で `pitch_seq_encoder_type: my_encoder` を指定するだけで利用可能。

### 新しい BatterHistEncoder を追加

`components/batter_hist_encoders.py` にクラスを追加し、レジストリに登録する。

```python
@register_batter_hist_encoder("my_encoder")
class MyBatterHistEncoder(BaseBatterHistEncoder):
    def __init__(self, cfg, num_cont, seq_pitch_type_embed, seq_swing_result_embed):
        super().__init__(cfg, num_cont, seq_pitch_type_embed, seq_swing_result_embed)
        ...
        self._output_dim = ...

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, hist_pitch_type, hist_cont, hist_swing_attempt,
                hist_swing_result, hist_bb_type, hist_launch_speed,
                hist_launch_angle, hist_spray_angle,
                hist_pitch_mask, hist_atbat_mask) -> torch.Tensor:
        ...
```

YAML で `batter_hist_encoder_type: my_encoder` を指定するだけで利用可能。

### 新しい Head Strategy を追加

`components/head_strategies.py` に追加し、`HEAD_STRATEGY_REGISTRY` に登録する。

```python
class MyHeadStrategy(nn.Module):
    def __init__(self, cfg: ModelConfig, backbone_out: int):
        ...

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "swing_attempt": ...,  # (B,) logits
            "swing_result": ...,   # (B, num_swing_result) logits
            "bb_type": ...,        # (B, num_bb_type) logits
            "regression": ...,     # (B, D) or dict
        }

HEAD_STRATEGY_REGISTRY["my_strategy"] = MyHeadStrategy
```

## 規約

| 項目 | 要件 |
|------|------|
| Backbone | `__init__(input_dim, hidden_dims, dropout)` / `output_dim` プロパティ / `forward(x) → Tensor` |
| HeadStrategy | `__init__(cfg, backbone_out)` / `forward(h) → dict`（`model_scope` に応じたキーを返す） |
| PitchSeqEncoder | `BasePitchSeqEncoder` 継承 / `output_dim` プロパティ / `forward(seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result, seq_mask) → Tensor` |
| BatterHistEncoder | `BaseBatterHistEncoder` 継承 / `output_dim` プロパティ / `forward(hist_pitch_type, hist_cont, ..., hist_pitch_mask, hist_atbat_mask) → Tensor` |
| 出力 dict keys | `model_scope` に応じたサブセット: `swing_attempt`, `swing_result`, `bb_type`, `regression` |
