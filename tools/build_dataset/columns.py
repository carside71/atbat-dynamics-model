"""カラム名定数・マッピング定義."""

# ---------------------------------------------------------------------------
# 生データから抽出するカラム
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    # 打撃結果
    "description",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    # 投球情報
    "p_throws",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    # 軌道特徴量
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    # ストライクゾーン
    "sz_top",
    "sz_bot",
    # 打者情報
    "batter",
    "stand",
    # ゲームコンテキスト
    "inning",
    "inning_topbot",
    "outs_when_up",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
    "pitch_number",
    # ゲーム情報（分割・履歴用）
    "game_pk",
    "game_date",
    # メタデータ用
    "pitcher",
    "home_team",
    "away_team",
    "at_bat_number",
]

# ---------------------------------------------------------------------------
# description → (swing_attempt, swing_result) マッピング
# ---------------------------------------------------------------------------
DESCRIPTION_MAP: dict[str, tuple[bool, str | None]] = {
    # スイングあり — ヒット
    "hit_into_play": (True, "hit_into_play"),
    # スイングあり — ファウル系
    "foul": (True, "foul"),
    "foul_tip": (True, "foul"),
    "foul_bunt": (True, "foul"),
    "foul_pitchout": (True, "foul"),
    "bunt_foul_tip": (True, "foul"),
    # スイングあり — 空振り系
    "swinging_strike": (True, "miss"),
    "swinging_strike_blocked": (True, "miss"),
    "missed_bunt": (True, "miss"),
    # スイングなし
    "ball": (False, None),
    "blocked_ball": (False, None),
    "automatic_ball": (False, None),
    "called_strike": (False, None),
    "automatic_strike": (False, None),
    "hit_by_pitch": (False, None),
    "pitchout": (False, None),
}

# swing_result のクラスラベル順（頻度降順）
SWING_RESULT_CLASSES = ["foul", "hit_into_play", "miss"]

# ---------------------------------------------------------------------------
# エンコード対象カテゴリカル特徴量
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["p_throws", "pitch_type", "batter", "stand"]

# ---------------------------------------------------------------------------
# スプレーアングル計算用定数
# ---------------------------------------------------------------------------
HC_X_CENTER = 125.42
HC_Y_CENTER = 198.27

# ---------------------------------------------------------------------------
# 打者フィルタ閾値
# ---------------------------------------------------------------------------
MIN_PITCHES = 2000

# ---------------------------------------------------------------------------
# 時系列分割境界
# ---------------------------------------------------------------------------
TRAIN_END = "2024-06-30"
VALID_END = "2024-10-31"

# ---------------------------------------------------------------------------
# 打者履歴
# ---------------------------------------------------------------------------
BATTER_HIST_NUM_ATBATS = 50
PITCHER_HIST_NUM_ATBATS = 50

# ---------------------------------------------------------------------------
# GT整合性チェック用定数
# ---------------------------------------------------------------------------
GT_CLS_COLS = ["swing_attempt", "swing_result", "bb_type"]
GT_REG_COLS = ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]

COOCCUR_PAIRS = [("hc_x", "hc_y"), ("launch_speed", "launch_angle")]

BB_TYPE_ANGLE_RANGES = {
    0: (-90.0, 80.0),   # ground_ball
    1: (10.0, 75.0),    # fly_ball
    2: (-15.0, 45.0),   # line_drive
    3: (20.0, 90.0),    # popup
}

PHYSICAL_BOUNDS = {
    "launch_speed": (0.0, 125.0),
    "launch_angle": (-90.0, 90.0),
    "hc_x": (0.0, 250.0),
    "hc_y": (0.0, 250.0),
}

FAIL_THRESHOLD = 0.001  # 0.1%
