library(xml2)
library(dplyr)
library(readr)
library(here)

# ── Config ────────────────────────────────────────────────────────────────────

URL_PATTERN <- "https://ratings.fide.com/download/standard_dec%sfrl_xml.zip"
OUT_FILE    <- here("data", "processed", "fide_top100_2020_2026.csv")
YEARS       <- 2020:2026

# ── Завантаження і парсинг одного файлу ──────────────────────────────────────

download_and_parse <- function(year) {

  yy  <- sprintf("%02d", year - 2000)
  url <- sprintf(URL_PATTERN, yy)
  message("  завантажую dec", yy, " ...")

  tryCatch({

    zip_path    <- tempfile(fileext = ".zip")
    extract_dir <- tempfile()          # унікальна директорія для кожного року
    dir.create(extract_dir)

    download.file(url, zip_path, mode = "wb", quiet = TRUE)

    # дізнаємось точну назву XML всередині архіву, потім розпаковуємо тільки її
    zip_contents <- unzip(zip_path, list = TRUE)
    xml_name     <- zip_contents$Name[grepl("\\.xml$", zip_contents$Name)][1]
    unzip(zip_path, files = xml_name, exdir = extract_dir)

    xml_file <- file.path(extract_dir, xml_name)
    doc      <- read_xml(xml_file)
    players  <- xml_find_all(doc, "//player")

    get_field <- function(nodes, tag) xml_text(xml_find_first(nodes, tag))

    data.frame(
      name       = get_field(players, "name"),
      federation = get_field(players, "country"),
      rating     = as.integer(get_field(players, "rating")),
      title      = get_field(players, "title")
    ) |>
      filter(!is.na(rating), rating > 0) |>
      arrange(desc(rating)) |>
      slice_head(n = 100) |>
      mutate(rank = row_number(), year = year, month = "DEC", .before = 1)

  }, error = function(e) {
    message("  [skip] dec", yy, " — ", conditionMessage(e))
    NULL
  })
}

# ── Запускаємо ────────────────────────────────────────────────────────────────

message("Завантажую ", length(YEARS), " XML файлів з ratings.fide.com...\n")

result <- lapply(YEARS, download_and_parse) |> bind_rows()

# ── Зберігаємо ───────────────────────────────────────────────────────────────

write_csv(result, OUT_FILE)

message("\nГотово! ", nrow(result), " рядків → ", OUT_FILE)
