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
    backbones.py       # DNNBackbone, ResDNNBackbone
    heads.py           # build_mlp_head(), MDNHead
    seq_encoders.py    # GRUSeqEncoder, TransformerSeqEncoder
    batter_history.py  # HierarchicalGRUBatterHistoryEncoder
    head_strategies.py # IndependentHeadStrategy, CascadeHeadStrategy
```

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
  <td style="background:#f3e5f5; border:2px solid #ab47bc; border-radius:8px; padding:10px 16px; text-align:center;"><b>SeqEncoder</b></td>
</tr>
<tr>
  <td style="background:#eceff1; border:2px solid #78909c; border-radius:8px; padding:6px 12px; text-align:center;">打者履歴 <i>(opt)</i></td>
  <td style="border:none; text-align:center; font-size:20px; color:#546e7a; padding:0 6px;">→</td>
  <td style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:10px 16px; text-align:center;"><b>BatterHistEncoder</b></td>
</tr>
</table>
</div>

### 出力（全構成共通）

| キー | 形状 | 内容 |
|---|---|---|
| `swing_attempt` | `(B,)` | スイング試行 logit (binary) |
| `swing_result` | `(B, num_swing_result)` | スイング結果 logits |
| `bb_type` | `(B, num_bb_type)` | 打球タイプ logits |
| `regression` | `(B, D)` or `dict` | launch_speed, launch_angle, hit_distance_sc, hc_x, hc_y（MDN の場合は `pi`, `mu`, `sigma` の dict）。D = `num_reg_targets`（デフォルト 5） |

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

YAML の `head_strategy` で選択。

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
  <td style="background:#fff3e0; border:2px solid #ffa726; border-radius:8px; padding:6px 14px; text-align:center;"><b>MLP / MDN</b></td>
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
- **`detach_cascade`**: `true` にすると上流ヘッドからの勾配を `detach` し、下流ヘッド学習時に上流を更新しない
- **設計意図**: スイング試行 → スイング結果 → 打球タイプ → 回帰値 という因果構造を反映

---

### 4. Regression Head Type

YAML の `regression_head_type` で選択。

#### `mlp`（デフォルト）

通常の MLP ヘッド。出力 `(B, D)`。D = `num_reg_targets`（デフォルト 5）。

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

---

### 5. Sequence Encoder (`components/seq_encoders.py`)

`max_seq_len > 0` のとき有効化。YAML の `seq_encoder_type` で選択。

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
  <td rowspan="4" style="background:#fffde7; border:2px solid #fdd835; border-radius:8px; padding:8px 14px; text-align:center; font-size:13px; vertical-align:middle;"><b>(B, seq_out_dim)</b></td>
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

- `Linear` で入力を `seq_hidden_dim` に射影後、`TransformerEncoder` で処理
- `src_key_padding_mask` でパディング位置をマスク
- マスク付き平均プーリングで固定長ベクトル化

---

### 6. Batter History Encoder (`components/batter_history.py`)

`batter_hist_max_atbats > 0` のとき有効化。`max_seq_len > 0`（SeqEncoder 有効）が前提。

打者の直近 N 打席の Statcast 全投球データを階層 GRU でエンコードする。

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
    <span style="background:#fafafa; border:2px dashed #9e9e9e; border-radius:8px; padding:6px 12px; text-align:center; font-size:13px; display:inline-block; padding:6px 14px;"><b>concat</b>: + bb_type_emb (4) + launch_speed (1) + launch_angle (1) + hc_x (1) + hc_y (1)</span>
  </td>
</tr>
<tr>
  <td colspan="3" style="border:none; text-align:center; color:#546e7a; font-size:14px; padding:4px 0;">↓ (B, N, D_inner + 8)</td>
</tr>
<tr>
  <td colspan="3" style="border:none; text-align:center; padding:2px 0;">
    <span style="background:#fce4ec; border:2px solid #ef5350; border-radius:8px; padding:10px 16px; text-align:center; display:inline-block; padding:8px 20px;"><b>Outer GRU</b> → h_n[-1] → <b>(B, D_hist_out)</b></span>
  </td>
</tr>
</table>
</div>

- **Inner GRU**: 各打席の投球列を 1 本の打席ベクトルに圧縮
- **Outer GRU**: N 打席分の打席ベクトル列を時系列処理して打者の傾向ベクトルに圧縮
- `pitch_type` / `swing_result` の Embedding は SeqEncoder と **重みを共有**
- 投球がない打席・履歴がない打者はゼロベクトル

---

## YAML 設定

### モデル設定フィールド一覧

| フィールド | デフォルト | 選択肢 | 説明 |
|---|---|---|---|
| `backbone_type` | `"resdnn"` | `"dnn"`, `"resdnn"` | Backbone の種類 |
| `backbone_hidden` | `[512, 256, 128]` | — | 各層の隠れ次元 |
| `dropout` | `0.2` | — | Dropout 率 |
| `head_strategy` | `"independent"` | `"independent"`, `"cascade"` | ヘッド接続戦略 |
| `head_hidden` | `[64]` | — | ヘッド MLP の隠れ次元 |
| `head_activation` | `"gelu"` | `"relu"`, `"gelu"` | ヘッド MLP の活性化関数 |
| `detach_cascade` | `true` | — | cascade 時に上流勾配を detach |
| `regression_head_type` | `"mlp"` | `"mlp"`, `"mdn"` | 回帰ヘッドの種類 |
| `mdn_num_components` | `5` | — | MDN のガウス成分数 |
| `max_seq_len` | `0` | — | 0: 無効、>0: 投球系列エンコーダ有効 |
| `seq_encoder_type` | `"gru"` | `"gru"`, `"transformer"` | 系列エンコーダの種類 |
| `seq_hidden_dim` | `64` | — | 系列エンコーダの隠れ次元 |
| `seq_num_layers` | `1` | — | 系列エンコーダの層数 |
| `seq_bidirectional` | `false` | — | GRU のみ: 双方向フラグ |
| `batter_hist_max_atbats` | `0` | — | 0: 無効、>0: 打者履歴エンコーダ有効 |
| `batter_hist_max_pitches` | `10` | — | 各打席の最大投球数 |
| `batter_hist_hidden_dim` | `64` | — | 打者履歴 GRU の隠れ次元 |
| `batter_hist_num_layers` | `1` | — | Outer GRU の層数 |

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

#### ResBlock DNN + GRU 系列エンコーダ

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  max_seq_len: 10
  seq_encoder_type: gru
  seq_hidden_dim: 64
  seq_num_layers: 1
  seq_bidirectional: false
```

#### ResBlock DNN + GRU 系列 + 打者履歴

```yaml
model:
  backbone_type: resdnn
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2
  max_seq_len: 10
  seq_encoder_type: gru
  seq_hidden_dim: 64
  seq_num_layers: 1
  seq_bidirectional: false
  batter_hist_max_atbats: 50
  batter_hist_max_pitches: 10
  batter_hist_hidden_dim: 64
  batter_hist_num_layers: 1
```

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

### 新しい Sequence Encoder を追加

`components/seq_encoders.py` にクラスを追加し、レジストリに登録する。

```python
@register_seq_encoder("my_encoder")
class MySeqEncoder(BaseSeqEncoder):
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

YAML で `seq_encoder_type: my_encoder` を指定するだけで利用可能。

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
| HeadStrategy | `__init__(cfg, backbone_out)` / `forward(h) → dict` |
| SeqEncoder | `BaseSeqEncoder` 継承 / `output_dim` プロパティ / `forward(seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result, seq_mask) → Tensor` |
| 出力 dict keys | `swing_attempt`, `swing_result`, `bb_type`, `regression` |
