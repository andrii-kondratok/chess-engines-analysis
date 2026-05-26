library(dplyr)
library(readr)
library(here)

# ── Читаємо файли ─────────────────────────────────────────────────────────────

olimpbase <- read_csv(here("data", "processed", "chess_elo_top100_until2001.csv"),
                      show_col_types = FALSE)

github    <- read_csv(here("data", "processed", "fide_top100_2001_2019.csv"),
                      show_col_types = FALSE)

fide_xml  <- read_csv(here("data", "processed", "fide_top100_2020_2026.csv"),
                      show_col_types = FALSE)

# ── Словник місяців (рядок → число) ──────────────────────────────────────────

month_lookup <- c(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
                  JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)

# ── Нормалізація кожного джерела ──────────────────────────────────────────────

# Файл 1: PascalCase, місяць числом, федерація — код FIDE (URS, USA...)
clean_olimpbase <- olimpbase |>
  rename(
    rank       = Rank,
    name       = Name,
    federation = Country,
    rating     = Rating,
    year       = Year,
    month      = Month
  ) |>
  mutate(
    # month=0 — це Unofficial January (1969, 1970): зберігалось як 0 в CSV
    month  = if_else(month == 0L, 1L, as.integer(month)),
    source = "olimpbase"
  ) |>
  filter(year < 2001) |>
  # Olimpbase містить кілька знімків на рік (Jan + Jul + інші).
  # Для аналізу тримаємо один на рік — беремо той що найближче до середини
  # року (липень пріоритетний, далі інший доступний місяць).
  group_by(year, month) |>
  mutate(rank = row_number()) |>   # перераховуємо ранг на випадок дублів
  ungroup() |>
  group_by(year) |>
  filter(month == if_else(7L %in% month, 7L, min(month))) |>
  ungroup()

# Файл 2: місяць — рядок (JUL), федерація — ПОВНА назва країни (Russia, India...)
# Увага: federation тут не сумісна з іншими джерелами — зберігаємо як є.
clean_github <- github |>
  mutate(
    month  = as.integer(month_lookup[month]),
    source = "fide_github"
  )

# Файл 3: місяць — рядок (DEC), федерація — 3-літерний код (NOR, IND...)
# Є зайва колонка title — прибираємо.
clean_fide_xml <- fide_xml |>
  mutate(
    month  = as.integer(month_lookup[month]),
    source = "fide_xml"
  ) |>
  select(-title)

# ── Об'єднуємо ────────────────────────────────────────────────────────────────
#
# ВАЖЛИВО: federation не однорідна між джерелами:
#   olimpbase   → FIDE/IOC код         (URS, RUS, USA)
#   fide_github → повна назва країни   (Russia, United States)
#   fide_xml    → 3-літерний код       (RUS, USA, NOR)
#
# Колонка залишається як є — нормалізація у окремому кроці якщо треба.

# XML поки виключаємо — потрібен перепарсинг (баг з дублюванням dec20)
combined <- bind_rows(clean_olimpbase, clean_github, clean_fide_xml) |>
  select(year, month, rank, name, federation, rating, source) |>
  arrange(year, month, rank)

# ── Базова перевірка ──────────────────────────────────────────────────────────

message("Рядків загалом: ", nrow(combined))
message("Діапазон років: ", min(combined$year), "–", max(combined$year))
message("\nРядків по джерелу:")
print(count(combined, source))

overlap <- combined |>
  distinct(year, month, source) |>
  count(year, month) |>
  filter(n > 1)

if (nrow(overlap) > 0) {
  message("\nПеретин дат між джерелами (", nrow(overlap), " місяців):")
  print(overlap)
} else {
  message("\nПеретинів між джерелами немає.")
}

# ── Зберігаємо ───────────────────────────────────────────────────────────────

write_csv(combined, here("data", "processed", "ratings_combined.csv"))
message("\nГотово → data/processed/ratings_combined.csv")
