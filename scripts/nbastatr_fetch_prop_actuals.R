#!/usr/bin/env Rscript

# Usage:
# Rscript scripts/nbastatr_fetch_prop_actuals.R --date 2025-01-15 --out data/processed/props_actuals_2025-01-15.csv
# or
# Rscript scripts/nbastatr_fetch_prop_actuals.R --start 2024-10-01 --end 2025-06-30 --out data/processed/props_actuals_2024-10-01_2025-06-30.csv

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(lubridate)
})

# Ensure nbastatR is available; try to guide installation otherwise
ensure_nbastatr <- function() {
  if (!requireNamespace("nbastatR", quietly = TRUE)) {
    message("Package 'nbastatR' not installed. Install with: remotes::install_github('abresler/nbastatR')")
    quit(status = 2)
  }
}

option_list <- list(
  make_option(c("--date"), type = "character", default = NULL, help = "Single date YYYY-MM-DD"),
  make_option(c("--start"), type = "character", default = NULL, help = "Start date YYYY-MM-DD"),
  make_option(c("--end"), type = "character", default = NULL, help = "End date YYYY-MM-DD"),
  make_option(c("--out"), type = "character", default = NULL, help = "Output CSV path")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (!is.null(opt$date) && (!is.null(opt$start) || !is.null(opt$end))) {
  stop("Provide either --date or --start/--end, not both.")
}

if (is.null(opt$date) && (is.null(opt$start) || is.null(opt$end))) {
  stop("Provide --date or both --start and --end")
}

ensure_nbastatr()

# Build date sequence
if (!is.null(opt$date)) {
  dates <- as.Date(opt$date)
} else {
  dates <- seq(as.Date(opt$start), as.Date(opt$end), by = "day")
}

# We'll use nbastatR::game_logs with seasons as end year; then filter by dateGame
suppressPackageStartupMessages({ library(nbastatR) })

collect_for_seasons <- function(seasons_end_years) {
  # returns tibble of player game logs for given seasons
  nbastatR::game_logs(
    seasons = seasons_end_years,
    result_types = c("player"),
    season_types = c("Regular Season", "Playoffs"),
    nest_data = FALSE,
    assign_to_environment = FALSE,
    return_message = FALSE
  )
}

# Determine unique seasons from dates
seasons_end <- unique(ifelse(month(dates) >= 7, year(dates) + 1, year(dates)))

logs <- suppressMessages(collect_for_seasons(seasons_end))

if (is.null(logs) || nrow(logs) == 0) {
  message("No logs returned from nbastatR")
  quit(status = 0)
}

# Normalize columns; nbastatR typical columns: dateGame, idGame, slugTeam, namePlayer, idPlayer, pts, treb, ast, fg3m
# PRA = pts + treb + ast
logs <- logs %>%
  mutate(date = as.Date(.data$dateGame)) %>%
  filter(.data$date %in% dates) %>%
  transmute(
    date = .data$date,
    game_id = .data$idGame,
    team = .data$slugTeam,
    player_id = .data$idPlayer,
    player_name = .data$namePlayer,
    pts = as.numeric(.data$pts),
    reb = as.numeric(.data$treb),
    ast = as.numeric(.data$ast),
    threes = as.numeric(.data$fg3m),
    pra = as.numeric(.data$pts) + as.numeric(.data$treb) + as.numeric(.data$ast)
  ) %>% distinct()

if (nrow(logs) == 0) {
  message("No rows matched the requested date(s)")
  quit(status = 0)
}

# If no --out provided, build a default path
if (is.null(opt$out)) {
  if (!is.null(opt$date)) {
    opt$out <- file.path("data", "processed", paste0("props_actuals_", opt$date, ".csv"))
  } else {
    opt$out <- file.path("data", "processed", paste0("props_actuals_", min(dates), "_", max(dates), ".csv"))
  }
}

dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)

readr::write_csv(logs, opt$out)
message(paste0("Wrote ", nrow(logs), " rows to ", opt$out))
