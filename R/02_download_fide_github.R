library(dplyr)
library(readr)
library(purrr)
library(here)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL <- paste0(
  "https://raw.githubusercontent.com/anujdahiya24/FIDE/main/",
  "Step%204%20-%20Cleaning/Cleaned%20csvs/"
)

OUT_FILE <- here("data", "processed", "fide_top100_2001_2019.csv")
SLEEP_SEC <- 1

# ── Файли для завантаження ────────────────────────────────────────────────────
#
# DEC існує тільки з 2012 по 2019.
# Для 2001–2011 беремо JUL — найближчий доступний щорічний знімок.

files <- bind_rows(
  data.frame(year = 2001:2011, month = "JUL", prefix = "JUL"),
  data.frame(year = 2012:2019, month = "DEC", prefix = "DEC")
) |>
  mutate(
    year_short = sprintf("%02d", year - 2000),
    filename   = paste0(prefix, year_short, ".csv"),
    url        = paste0(BASE_URL, filename)
  )

# ── Завантаження одного файлу ─────────────────────────────────────────────────

download_one <- function(url, filename, year, month) {

  Sys.sleep(SLEEP_SEC)
  message("  завантажую ", filename, " ...")

  tryCatch({

    raw <- read_delim(url, delim = "*", show_col_types = FALSE)

    raw |>
      select(name = Name, federation = Country, rating = Rating) |>
      filter(!is.na(rating)) |>
      arrange(desc(rating)) |>
      slice_head(n = 100) |>
      mutate(
        rank       = row_number(),
        year       = year,
        month      = month,
        .before    = 1
      )

  }, error = function(e) {
    message("  [skip] ", filename, " — ", conditionMessage(e))
    NULL
  })
}

# ── Запускаємо ────────────────────────────────────────────────────────────────

message("Завантажую ", nrow(files), " файлів з GitHub...\n")

result <- pmap(
  list(files$url, files$filename, files$year, files$month),
  download_one
) |>
  list_rbind()

# ── Зберігаємо ───────────────────────────────────────────────────────────────

write_csv(result, OUT_FILE)

message("\nГотово! ", nrow(result), " рядків → ", OUT_FILE)
