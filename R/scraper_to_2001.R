library(rvest)
library(dplyr)
library(stringr)
library(readr)

# 1. Собираем только нужные ссылки
library(rvest)
library(stringr)
library(dplyr)

# 1. Проста функція збору всіх посилань
get_all_links <- function() {
  url <- "https://www.olimpbase.org/Elo/"
  page <- read_html(url)
  
  # Витягуємо всі посилання (теги <a>)
  links <- page %>% html_nodes("a") %>% html_attr("href")
  
  # Фільтруємо лише файли рейтингів (починаються на Elo/elo і мають цифри)
  elo_files <- links[str_detect(links, "(?i)^elo[0-9]{6}.*\\.html$")]
  
  # Формуємо повні URL
  full_urls <- paste0("https://www.olimpbase.org/Elo/", elo_files)
  return(full_urls)
}

# 2. Отримуємо сирий список
raw_links <- get_all_links()

# 3. Фільтруємо окремо: залишаємо ТІЛЬКИ ті, що мають 'e' перед .html
# Або просто ті, де є цифри і закінчуються на e.html
filtered_links <- raw_links[str_detect(raw_links, "(?i)e\\.html$")]

# 2. Парсер (остается прежним, так как структура <pre> там одинаковая)
parse_elo_page <- function(url) {
  page <- tryCatch(read_html(url), error = function(e) return(NULL))
  if (is.null(page)) return(NULL)
  
  pre_node <- page %>% html_node("pre")
  if (is.na(pre_node)) return(NULL)
  
  raw_text <- html_text(pre_node)
  lines <- unlist(str_split(raw_text, "\n"))
  data_start <- which(str_detect(lines, "^\\s*[0-9]+"))
  if (length(data_start) == 0) return(NULL)
  
  data_lines <- lines[min(data_start):length(lines)]
  data_lines <- data_lines[nchar(str_trim(data_lines)) > 20]
  
  widths <- fwf_cols(
    Rank = c(1, 6),
    ID = c(7, 16),
    Name = c(17, 58),
    Title = c(59, 62),
    Country = c(63, 67),
    Rating = c(68, 73)
  )
  
  df <- read_fwf(I(data_lines), col_positions = widths, show_col_types = FALSE)
  file_info <- str_extract(url, "[0-9]{6}")
  
  return(df %>%
           mutate(
             Rank = as.numeric(Rank),
             Rating = as.numeric(Rating),
             Year = as.numeric(substr(file_info, 1, 4)),
             Month = as.numeric(substr(file_info, 5, 6))
           ) %>%
           filter(!is.na(Rank)) %>%
           arrange(Rank) %>% # Дополнительная сортировка для верности
           head(100))
}

# --- ЗАПУСК ---
final_list <- list()
for (i in seq_along(filtered_links)) {
  message(sprintf("[%d/%d] Парсинг: %s", i, length(filtered_links), filtered_links[i]))
  res <- parse_elo_page(filtered_links[i])
  if (!is.null(res)) final_list[[i]] <- res
  Sys.sleep(0.2)
}

final_df <- bind_rows(final_list) %>% arrange(Year, Month, Rank)

unique(final_df$Year)

final_df <- final_df %>%
  select(-c(ID, Title))

write_csv(final_df, "chess_elo_top100_until2001.csv")
