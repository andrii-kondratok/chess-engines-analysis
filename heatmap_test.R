library(arrow)
library(tidyverse)
setwd("D:/навчання/R/chess_engines_analysis")
# Читаємо паркет з amateur + time фічами (найповніший)
df <- read_parquet("data/processed/lichess_amateur_v2.parquet")

# Відбираємо числові ознаки для кореляції
cor_features <- df %>%
select(
  avg_elo,
  white_acpl, black_acpl,
  white_acpl_opening, black_acpl_opening,
  white_acpl_middle, black_acpl_middle,
  theory_depth_swing,
  total_moves,
  time_control_seconds,
  white_opening_blunders, black_opening_blunders,
  white_pointless_checks, black_pointless_checks,
  white_castled, black_castled,
  white_avg_think, black_avg_think,
  white_think_std, black_think_std,
  white_fast_moves_pct, black_fast_moves_pct,
  white_opening_thinking, black_opening_thinking
)

# Матриця кореляцій (pairwise — ігнорує NA)
cor_mat <- cor(cor_features, use = "pairwise.complete.obs")

cor_long <- pivot_longer(
  rownames_to_column(as.data.frame(cor_mat), "var1"),
  -var1, names_to = "var2", values_to = "corr"
)

feat_order <- names(sort(cor_mat["avg_elo", ], decreasing = TRUE))

cor_long$var1 <- factor(cor_long$var1, levels = feat_order)
cor_long$var2 <- factor(cor_long$var2, levels = rev(feat_order))

ggplot(cor_long, aes(var1, var2, fill = corr)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", corr)), size = 2.6, color = "grey20") +
  scale_fill_gradient2(
    low = "#2E86AB", mid = "white", high = "#A23B72",
    midpoint = 0, limits = c(-1, 1), name = "Кореляція"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  ) +
  coord_fixed()
