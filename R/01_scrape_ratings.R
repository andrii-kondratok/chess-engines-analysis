library(rvest)
library(dplyr)
library(purrr)
library(readr)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL    <- "https://www.olimpbase.org/Elo/Elo%d%se.html"
OUT_RAW     <- "data/processed/top25_all_lists.csv"
OUT_METRICS <- "data/processed/convergence_metrics.csv"

TOP_N      <- 100   # players to keep per list
SLEEP_SEC  <- 2   # pause between requests — be polite

# ── Build the list of dates to scrape ─────────────────────────────────────────
#
# Просто пробуємо кожен місяць з 1967 по сьогодні.
# Якщо сторінки немає — tryCatch її пропустить, нічого не додається.

dates <- data.frame(
  date = seq.Date(as.Date("1967-01-01"), Sys.Date(), by = "month")
) |>
  mutate(
    date_str = format(date, "%Y-%m-%d"),
    url      = sprintf(BASE_URL, as.integer(format(date, "%Y")), format(date, "%m"))
  )


# ── Парсери ───────────────────────────────────────────────────────────────────

# Офіційні списки (1971+): стандартна HTML-таблиця
parse_table <- function(page) {
  tables <- html_elements(page, "table")
  if (length(tables) == 0) return(NULL)

  raw <- html_table(tables[[1]], fill = TRUE)

  colnames(raw)[1:min(ncol(raw), 11)] <- c(
    "rank", "fide_id", "name", "title", "federation", "rating",
    "change", "games", "birthday", "sex", "flag"
  )[1:min(ncol(raw), 11)]

  raw |>
    filter(suppressWarnings(!is.na(as.integer(rank)))) |>
    mutate(rank = as.integer(rank), rating = as.integer(rating)) |>
    select(rank, name, federation, rating)
}

# Неофіційні ранні списки (1967–1971): <pre> з fixed-width текстом
# Формат рядка:  "   1   [Fischer, Robert James]   USA  2760"
parse_pre <- function(page) {
  pre <- html_elements(page, "pre")
  if (length(pre) == 0) return(NULL)

  lines <- strsplit(html_text(pre[[1]]), "\n")[[1]]

  matches <- regmatches(lines, regexec(
    "^\\s*(\\d+[=]?)\\s+\\[([^\\]]+)\\]\\s+([A-Z]{3})\\s+(\\d{4})",
    lines
  ))

  rows <- Filter(\(m) length(m) == 5, matches)
  if (length(rows) == 0) return(NULL)

  data.frame(
    rank       = as.integer(gsub("=", "", sapply(rows, `[`, 2))),
    name       = sapply(rows, `[`, 3),
    federation = sapply(rows, `[`, 4),
    rating     = as.integer(sapply(rows, `[`, 5))
  )
}

# ── Scrape a single list ───────────────────────────────────────────────────────

scrape_one_list <- function(url, date_str) {

  Sys.sleep(SLEEP_SEC)

  tryCatch({

    page <- read_html(url)

    parsed <- parse_table(page)
    if (is.null(parsed)) parsed <- parse_pre(page)
    if (is.null(parsed)) return(NULL)

    parsed |>
      head(TOP_N) |>
      mutate(date = date_str, .before = 1)

  }, error = function(e) {
    message("  [skip] ", date_str, " — ", conditionMessage(e))
    NULL
  })
}


# ── Run the scraper ────────────────────────────────────────────────────────────

message("Scraping ", nrow(dates), " rating lists from olimpbase.org ...")
message("Estimated time: ~", round(nrow(dates) * SLEEP_SEC / 60, 1), " minutes\n")

top100_raw <- map2(
  dates$url,
  dates$date_str,
  \(url, d) {
    message("  fetching ", d, " ...")
    scrape_one_list(url, d)
  }
) |>
  list_rbind()


# ── Compute convergence metrics ────────────────────────────────────────────────
#
# Key question: has the gap between #1 and the pack shrunk over time?

convergence_metrics <- top25_raw |>
  group_by(date) |>
  summarise(
    n_scraped         = n(),
    rating_no1        = first(rating[rank == 1]),
    rating_mean_5_10  = mean(rating[rank %in% 5:10],  na.rm = TRUE),
    rating_mean_15_20 = mean(rating[rank %in% 15:20], na.rm = TRUE),
    gap_no1_to_15_20  = rating_no1 - rating_mean_15_20,
    sd_top20          = sd(rating[rank <= 20],         na.rm = TRUE),
    .groups = "drop"
  )


# ── Save ───────────────────────────────────────────────────────────────────────

write_csv(top25_raw,          OUT_RAW)
write_csv(convergence_metrics, OUT_METRICS)

message("\nDone!")
message("  Raw data : ", nrow(top25_raw), " rows → ", OUT_RAW)
message("  Metrics  : ", nrow(convergence_metrics), " lists → ", OUT_METRICS)
