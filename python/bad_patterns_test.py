import pandas as pd
df = pd.read_parquet('data/processed/lichess_amateur.parquet')

# Чи різниться поведінка по рейтингу?
df['bucket'] = pd.cut(df['avg_elo'], bins=[800, 1200, 1600, 2000, 2200])
new_features = [
    'white_pointless_checks', 'white_early_queen_blunders',
    'white_opening_blunders', 'white_castled',
    'white_pawn_moves_in_opening', 'white_piece_moved_twice_opening'
]
print(df.groupby('bucket')[new_features].mean().round(2))

new_features_extended = [
    'white_bad_corner_bishop', 'white_bad_rim_knight',
    'white_castled_move'
]
print(df.groupby('bucket')[new_features_extended].mean().round(2))