#!/usr/bin/env Rscript
# ── filter_duplicate_masks.R ─────────────────────────────────────────────────
#
# Filters a combined_morphology_results.csv to remove rows where different
# target areas (area_um2) produced identical masks for the same cell.
#
# The mask generation algorithm enforces that every mask is at least as large
# as the soma, and growth can hit an intensity floor. This means multiple
# target area levels can produce pixel-identical masks. This script detects
# those duplicates by grouping on (image_name, soma_id) and comparing
# mask_area values across target sizes. When consecutive target areas yield
# the same actual mask_area, only the row whose target area (area_um2) is
# closest to the measured mask_area is kept.
#
# Usage:
#   Rscript filter_duplicate_masks.R <input_csv> [output_csv]
#
#   If output_csv is omitted, writes to <input_basename>_filtered.csv
#
# Example:
#   Rscript filter_duplicate_masks.R combined_morphology_results.csv
#   Rscript filter_duplicate_masks.R combined_morphology_results.csv cleaned.csv
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required. Install with: install.packages('dplyr')")
  }
  library(dplyr)
})

# ── Parse arguments ──────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  # Interactive mode: use file chooser
  if (interactive()) {
    input_csv <- file.choose()
  } else {
    cat("Usage: Rscript filter_duplicate_masks.R <input_csv> [output_csv]\n")
    quit(status = 1)
  }
} else {
  input_csv <- args[1]
}

if (!file.exists(input_csv)) {
  stop(paste("File not found:", input_csv))
}

if (length(args) >= 2) {
  output_csv <- args[2]
} else {
  base <- tools::file_path_sans_ext(input_csv)
  output_csv <- paste0(base, "_filtered.csv")
}

# ── Read data ────────────────────────────────────────────────────────────────

df <- read.csv(input_csv, stringsAsFactors = FALSE)

cat(sprintf("Read %d rows from %s\n", nrow(df), basename(input_csv)))
cat(sprintf("  Unique images:  %d\n", length(unique(df$image_name))))
cat(sprintf("  Unique somas:   %d\n", nrow(distinct(df, image_name, soma_id))))
cat(sprintf("  Target areas:   %s\n",
            paste(sort(unique(df$area_um2)), collapse = ", ")))

# ── Identify and remove duplicate masks ──────────────────────────────────────
#
# For each cell (image_name + soma_id), sort by target area and find runs of
# identical mask_area. Within each run, keep only the row whose area_um2
# best matches the actual mask_area.

filter_duplicate_masks <- function(df, tolerance = 0.01) {
  # Validate required columns
  required <- c("image_name", "soma_id", "area_um2", "mask_area")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop(paste("Missing required columns:", paste(missing, collapse = ", ")))
  }

  df %>%
    group_by(image_name, soma_id) %>%
    arrange(area_um2, .by_group = TRUE) %>%
    mutate(
      # Detect when mask_area changes between consecutive target sizes
      # Use relative tolerance to handle floating point differences
      mask_area_changed = if (n() == 1) TRUE else {
        c(TRUE, abs(diff(mask_area)) / pmax(mask_area[-n()], 1e-9) > tolerance)
      },
      # Assign a group ID to each run of identical mask_area values
      mask_group = cumsum(mask_area_changed),
      # Within each run, pick the row whose target area is closest to actual
      area_diff = abs(area_um2 - mask_area)
    ) %>%
    group_by(image_name, soma_id, mask_group) %>%
    # Keep the row with the best-matching target area
    filter(area_diff == min(area_diff)) %>%
    # If there's still a tie, keep the smallest target area
    filter(area_um2 == min(area_um2)) %>%
    ungroup() %>%
    select(-mask_area_changed, -mask_group, -area_diff) %>%
    arrange(image_name, soma_id, area_um2)
}

filtered <- filter_duplicate_masks(df)

# ── Report results ───────────────────────────────────────────────────────────

n_removed <- nrow(df) - nrow(filtered)
cat(sprintf("\nFiltering results:\n"))
cat(sprintf("  Rows before:  %d\n", nrow(df)))
cat(sprintf("  Rows after:   %d\n", nrow(filtered)))
cat(sprintf("  Duplicates removed: %d (%.1f%%)\n",
            n_removed, 100 * n_removed / max(nrow(df), 1)))

# Per-cell summary of what was removed
if (n_removed > 0) {
  cat("\nPer-cell breakdown of removed duplicates:\n")

  merged <- df %>%
    left_join(
      filtered %>% mutate(.kept = TRUE),
      by = names(df)
    ) %>%
    mutate(.kept = !is.na(.kept))

  removed_detail <- merged %>%
    filter(!.kept) %>%
    group_by(image_name, soma_id) %>%
    summarise(
      n_removed = n(),
      removed_areas = paste(area_um2, collapse = ", "),
      mask_area_values = paste(unique(round(mask_area, 1)), collapse = ", "),
      .groups = "drop"
    ) %>%
    arrange(desc(n_removed))

  # Show up to 20 examples
  to_show <- head(removed_detail, 20)
  for (i in seq_len(nrow(to_show))) {
    row <- to_show[i, ]
    cat(sprintf("  %s | %s: removed %d rows (target areas: %s; actual mask_area: %s)\n",
                row$image_name, row$soma_id, row$n_removed,
                row$removed_areas, row$mask_area_values))
  }
  if (nrow(removed_detail) > 20) {
    cat(sprintf("  ... and %d more cells with duplicates\n",
                nrow(removed_detail) - 20))
  }
}

# ── Write output ─────────────────────────────────────────────────────────────

write.csv(filtered, output_csv, row.names = FALSE)
cat(sprintf("\nFiltered CSV written to: %s\n", output_csv))
