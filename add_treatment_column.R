# Add a "treatment" column derived from the first word of image_name
# Usage: Rscript add_treatment_column.R

library(readr)
library(dplyr)
library(stringr)

# ── Read the CSV ──────────────────────────────────────────────────────
csv_path <- file.choose()  # opens file picker
df <- read_csv(csv_path, show_col_types = FALSE)

# ── Extract treatment = first word of image_name ──────────────────────
df <- df %>%
  mutate(treatment = str_extract(image_name, "^[^_]+"))

# ── Move treatment to the front ──────────────────────────────────────
df <- df %>%
  select(treatment, everything())

# ── Write output ─────────────────────────────────────────────────────
out_path <- sub("\\.csv$", "_with_treatment.csv", csv_path)
write_csv(df, out_path)

cat("Done!\n")
cat("  Treatment groups found:", paste(unique(df$treatment), collapse = ", "), "\n")
cat("  Output:", out_path, "\n")
