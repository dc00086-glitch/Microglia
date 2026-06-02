#!/usr/bin/env Rscript
# ============================================================================
# MMPS Morphological Phenotype Classifier
#
# Classifies microglia into morphological phenotypes based on existing
# morphology metrics from the MMPS combined_morphology_results.csv
#
# Phenotypes: Ramified, Hypertrophic, Bushy, Amoeboid, Rod
#
# Usage:
#   Rscript mmps_phenotype_classifier.R combined_morphology_results.csv
#   Or open in RStudio and run interactively
# ============================================================================

# --- Install packages if needed ---
if (!require("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!require("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!require("tidyr", quietly = TRUE)) install.packages("tidyr")
if (!require("ggpubr", quietly = TRUE)) install.packages("ggpubr")

library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)

# ============================================================================
# LOAD DATA
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  csv_path <- args[1]
} else {
  csv_path <- "combined_morphology_results.csv"
}

if (!file.exists(csv_path)) {
  cat("ERROR: File not found:", csv_path, "\n")
  cat("Usage: Rscript mmps_phenotype_classifier.R <path_to_csv>\n")
  quit(status = 1)
}

data <- read.csv(csv_path, stringsAsFactors = FALSE)
cat("Loaded", nrow(data), "cells from", length(unique(data$image_name)), "images\n")

# ============================================================================
# CLASSIFICATION THRESHOLDS
#
# Adjust these based on your data and biological context.
# Run once with defaults, check the distribution, then tune.
# ============================================================================

thresholds <- list(
  # Amoeboid: soma dominates the cell (phagocytic, few/no processes)
  amoeboid_soma_ratio    = 0.55,

  # Bushy: enlarged soma with short thick retracted processes
  bushy_soma_ratio       = 0.30,
  bushy_roundness        = 0.35,

  # Rod: highly elongated bipolar cell
  rod_roundness          = 0.15,
  rod_eccentricity       = 0.85,

  # Hypertrophic: enlarged soma, thickened processes, early activation
  hypertrophic_soma_ratio = 0.20,
  hypertrophic_ti_max     = 5.0
)

cat("\n--- Classification Thresholds ---\n")
for (name in names(thresholds)) {
  cat(sprintf("  %-30s = %.2f\n", name, thresholds[[name]]))
}
cat("\n")

# ============================================================================
# COMPUTE DERIVED METRICS
# ============================================================================

# Check required columns
required <- c("soma_area", "mask_area", "perimeter", "roundness", "eccentricity")
missing <- required[!required %in% names(data)]
if (length(missing) > 0) {
  cat("ERROR: Missing required columns:", paste(missing, collapse = ", "), "\n")
  cat("Available columns:", paste(names(data), collapse = ", "), "\n")
  quit(status = 1)
}

data <- data %>%
  mutate(
    soma_cell_ratio = ifelse(mask_area > 0, soma_area / mask_area, 0),
    transformation_index = ifelse(mask_area > 0, (perimeter^2) / (4 * pi * mask_area), 0)
  )

cat("Derived metrics computed:\n")
cat(sprintf("  soma_cell_ratio:      mean=%.3f, sd=%.3f, range=[%.3f, %.3f]\n",
    mean(data$soma_cell_ratio, na.rm=TRUE), sd(data$soma_cell_ratio, na.rm=TRUE),
    min(data$soma_cell_ratio, na.rm=TRUE), max(data$soma_cell_ratio, na.rm=TRUE)))
cat(sprintf("  transformation_index: mean=%.2f, sd=%.2f, range=[%.2f, %.2f]\n",
    mean(data$transformation_index, na.rm=TRUE), sd(data$transformation_index, na.rm=TRUE),
    min(data$transformation_index, na.rm=TRUE), max(data$transformation_index, na.rm=TRUE)))

# ============================================================================
# CLASSIFY PHENOTYPES
#
# Priority order (first match wins):
#   1. Amoeboid  — soma_cell_ratio > 0.55
#   2. Bushy     — soma_cell_ratio > 0.30 AND roundness > 0.35
#   3. Rod       — roundness < 0.15 AND eccentricity > 0.85
#   4. Hypertrophic — soma_cell_ratio > 0.20 AND transformation_index < 5
#   5. Ramified  — everything else (homeostatic)
# ============================================================================

classify_phenotype <- function(soma_cell_ratio, roundness, eccentricity, transformation_index) {
  case_when(
    soma_cell_ratio > thresholds$amoeboid_soma_ratio ~ "Amoeboid",
    soma_cell_ratio > thresholds$bushy_soma_ratio &
      roundness > thresholds$bushy_roundness ~ "Bushy",
    roundness < thresholds$rod_roundness &
      eccentricity > thresholds$rod_eccentricity ~ "Rod",
    soma_cell_ratio > thresholds$hypertrophic_soma_ratio &
      transformation_index < thresholds$hypertrophic_ti_max ~ "Hypertrophic",
    TRUE ~ "Ramified"
  )
}

data <- data %>%
  mutate(
    phenotype = classify_phenotype(soma_cell_ratio, roundness, eccentricity, transformation_index),
    phenotype = factor(phenotype, levels = c("Ramified", "Hypertrophic", "Bushy", "Amoeboid", "Rod"))
  )

# ============================================================================
# SUMMARY TABLES
# ============================================================================

cat("\n============================================\n")
cat("PHENOTYPE CLASSIFICATION RESULTS\n")
cat("============================================\n\n")

# Overall counts
overall <- data %>%
  count(phenotype) %>%
  mutate(percent = round(n / sum(n) * 100, 1))

cat("Overall distribution:\n")
for (i in seq_len(nrow(overall))) {
  cat(sprintf("  %-15s %5d cells  (%5.1f%%)\n",
      overall$phenotype[i], overall$n[i], overall$percent[i]))
}
cat(sprintf("  %-15s %5d cells\n", "TOTAL", sum(overall$n)))

# Per-treatment if available
has_treatment <- "treatment" %in% names(data)
if (has_treatment) {
  data$treatment <- factor(data$treatment)
  n_groups <- nlevels(data$treatment)

  cat("\nPer-treatment distribution:\n")
  treatment_summary <- data %>%
    group_by(treatment, phenotype) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(treatment) %>%
    mutate(
      total = sum(n),
      percent = round(n / total * 100, 1)
    ) %>%
    ungroup()

  for (trt in levels(data$treatment)) {
    trt_data <- treatment_summary %>% filter(treatment == trt)
    total <- trt_data$total[1]
    cat(sprintf("\n  %s (n=%d):\n", trt, total))
    for (j in seq_len(nrow(trt_data))) {
      cat(sprintf("    %-15s %4d  (%5.1f%%)\n",
          trt_data$phenotype[j], trt_data$n[j], trt_data$percent[j]))
    }
  }

  # Percent phenotype per image (for graphing)
  per_image <- data %>%
    group_by(image_name, treatment, phenotype) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(image_name, treatment) %>%
    mutate(
      total = sum(n),
      percent = n / total * 100
    ) %>%
    ungroup()
}

# Per-image counts
per_image_wide <- data %>%
  group_by(image_name) %>%
  mutate(total = n()) %>%
  group_by(image_name, phenotype, total) %>%
  summarise(n = n(), .groups = "drop") %>%
  mutate(percent = round(n / total * 100, 1))

# Metric summary per phenotype
cat("\n\nMetric means by phenotype:\n")
cat(sprintf("%-15s %8s %8s %8s %8s %8s %8s\n",
    "Phenotype", "SCR", "TI", "Round", "Eccen", "Soma", "MaskArea"))
cat(paste(rep("-", 75), collapse=""), "\n")

metric_summary <- data %>%
  group_by(phenotype) %>%
  summarise(
    n = n(),
    soma_cell_ratio_mean = mean(soma_cell_ratio, na.rm = TRUE),
    transformation_index_mean = mean(transformation_index, na.rm = TRUE),
    roundness_mean = mean(roundness, na.rm = TRUE),
    eccentricity_mean = mean(eccentricity, na.rm = TRUE),
    soma_area_mean = mean(soma_area, na.rm = TRUE),
    mask_area_mean = mean(mask_area, na.rm = TRUE),
    .groups = "drop"
  )

for (i in seq_len(nrow(metric_summary))) {
  cat(sprintf("%-15s %8.3f %8.2f %8.3f %8.3f %8.1f %8.1f\n",
      metric_summary$phenotype[i],
      metric_summary$soma_cell_ratio_mean[i],
      metric_summary$transformation_index_mean[i],
      metric_summary$roundness_mean[i],
      metric_summary$eccentricity_mean[i],
      metric_summary$soma_area_mean[i],
      metric_summary$mask_area_mean[i]))
}

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Add phenotype and derived metrics to the original data and save
output_path <- sub("\\.csv$", "_with_phenotypes.csv", csv_path)
write.csv(data, output_path, row.names = FALSE)
cat("\n\nResults saved to:", output_path, "\n")

# Save summary tables
summary_path <- sub("\\.csv$", "_phenotype_summary.csv", csv_path)
if (has_treatment) {
  summary_wide <- data %>%
    group_by(image_name, treatment, phenotype) %>%
    summarise(n = n(), .groups = "drop") %>%
    pivot_wider(names_from = phenotype, values_from = n, values_fill = 0) %>%
    mutate(total = rowSums(across(c(Ramified, Hypertrophic, Bushy, Amoeboid, Rod)))) %>%
    mutate(across(c(Ramified, Hypertrophic, Bushy, Amoeboid, Rod),
                  ~ round(. / total * 100, 1),
                  .names = "pct_{.col}"))
} else {
  summary_wide <- data %>%
    group_by(image_name, phenotype) %>%
    summarise(n = n(), .groups = "drop") %>%
    pivot_wider(names_from = phenotype, values_from = n, values_fill = 0) %>%
    mutate(total = rowSums(across(c(Ramified, Hypertrophic, Bushy, Amoeboid, Rod)))) %>%
    mutate(across(c(Ramified, Hypertrophic, Bushy, Amoeboid, Rod),
                  ~ round(. / total * 100, 1),
                  .names = "pct_{.col}"))
}
write.csv(summary_wide, summary_path, row.names = FALSE)
cat("Summary table saved to:", summary_path, "\n")

# ============================================================================
# PLOTS
# ============================================================================

theme_pub <- theme_classic(base_size = 14) +
  theme(
    axis.title = element_text(face = "bold"),
    legend.position = "top",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

pheno_colors <- c(
  "Ramified"     = "#2196F3",  # blue
  "Hypertrophic" = "#FF9800",  # orange
  "Bushy"        = "#F44336",  # red
  "Amoeboid"     = "#9C27B0",  # purple
  "Rod"          = "#4CAF50"   # green
)

# --- 1. Overall phenotype distribution (pie chart) ---
p_pie <- ggplot(overall, aes(x = "", y = n, fill = phenotype)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  scale_fill_manual(values = pheno_colors) +
  labs(title = "Overall Phenotype Distribution", fill = "Phenotype") +
  theme_void(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("plot_phenotype_pie.png", p_pie, width = 7, height = 6, dpi = 300)

# --- 2. Phenotype bar chart ---
p_bar <- ggplot(overall, aes(x = phenotype, y = n, fill = phenotype)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = paste0(n, "\n(", percent, "%)")), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = pheno_colors) +
  labs(title = "Phenotype Counts", y = "Number of Cells", x = "") +
  theme_pub +
  theme(legend.position = "none") +
  ylim(0, max(overall$n) * 1.15)
ggsave("plot_phenotype_bar.png", p_bar, width = 8, height = 6, dpi = 300)

# --- 3. Per-treatment stacked bar (if treatment data exists) ---
if (has_treatment) {
  p_stacked <- ggplot(per_image, aes(x = treatment, y = percent, fill = phenotype)) +
    geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7, outlier.shape = NA) +
    geom_jitter(aes(group = phenotype), position = position_dodge(width = 0.8),
                size = 1.5, alpha = 0.5) +
    scale_fill_manual(values = pheno_colors) +
    labs(title = "Phenotype Distribution by Treatment",
         y = "% of Cells per Image", x = "", fill = "Phenotype") +
    theme_pub
  ggsave("plot_phenotype_by_treatment.png", p_stacked, width = 10, height = 6, dpi = 300)

  # Stacked percent bar per treatment
  trt_pct <- data %>%
    group_by(treatment, phenotype) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(treatment) %>%
    mutate(percent = n / sum(n) * 100) %>%
    ungroup()

  p_stacked_bar <- ggplot(trt_pct, aes(x = treatment, y = percent, fill = phenotype)) +
    geom_col(position = "stack", alpha = 0.85) +
    scale_fill_manual(values = pheno_colors) +
    labs(title = "Phenotype Composition by Treatment",
         y = "% of Cells", x = "", fill = "Phenotype") +
    theme_pub
  ggsave("plot_phenotype_stacked.png", p_stacked_bar, width = 8, height = 6, dpi = 300)

  # --- Statistical test: is phenotype distribution different between treatments? ---
  cat("\n--- Statistical Tests ---\n")

  if (n_groups == 2) {
    cat("Chi-squared test (2 treatments):\n")
    contingency <- table(data$treatment, data$phenotype)
    chi_test <- chisq.test(contingency)
    cat(sprintf("  X² = %.2f, df = %d, p = %s\n",
        chi_test$statistic, chi_test$parameter,
        format.pval(chi_test$p.value, digits = 4)))
    if (chi_test$p.value < 0.05) {
      cat("  * Phenotype distribution is significantly different between treatments\n")
    } else {
      cat("  Phenotype distribution is not significantly different\n")
    }

    # Per-phenotype Fisher tests
    cat("\nPer-phenotype comparisons (Fisher's exact test):\n")
    for (pheno in levels(data$phenotype)) {
      data$is_pheno <- data$phenotype == pheno
      tbl <- table(data$treatment, data$is_pheno)
      if (all(dim(tbl) == c(2, 2))) {
        ft <- fisher.test(tbl)
        sig <- ifelse(ft$p.value < 0.001, "***",
               ifelse(ft$p.value < 0.01, "**",
               ifelse(ft$p.value < 0.05, "*", "ns")))
        cat(sprintf("  %-15s p = %s  OR = %.2f  %s\n",
            pheno, format.pval(ft$p.value, digits = 4),
            ft$estimate, sig))
      }
    }
    data$is_pheno <- NULL
  } else if (n_groups >= 3) {
    cat("Chi-squared test (3+ treatments):\n")
    contingency <- table(data$treatment, data$phenotype)
    chi_test <- chisq.test(contingency)
    cat(sprintf("  X² = %.2f, df = %d, p = %s\n",
        chi_test$statistic, chi_test$parameter,
        format.pval(chi_test$p.value, digits = 4)))
    if (chi_test$p.value < 0.05) {
      cat("  * Phenotype distribution is significantly different across treatments\n")
    }

    # Pairwise Fisher tests for each phenotype
    cat("\nPairwise comparisons per phenotype:\n")
    pairs <- combn(levels(data$treatment), 2, simplify = FALSE)
    for (pheno in levels(data$phenotype)) {
      cat(sprintf("\n  %s:\n", pheno))
      for (pair in pairs) {
        sub <- data %>% filter(treatment %in% pair)
        sub$is_pheno <- sub$phenotype == pheno
        tbl <- table(factor(sub$treatment, levels = pair), sub$is_pheno)
        if (all(dim(tbl) == c(2, 2))) {
          ft <- fisher.test(tbl)
          sig <- ifelse(ft$p.value < 0.001, "***",
                 ifelse(ft$p.value < 0.01, "**",
                 ifelse(ft$p.value < 0.05, "*", "ns")))
          cat(sprintf("    %s vs %s: p = %s  %s\n",
              pair[1], pair[2], format.pval(ft$p.value, digits = 4), sig))
        }
      }
    }
  }
}

# --- 4. Scatter: soma_cell_ratio vs transformation_index colored by phenotype ---
p_scatter <- ggplot(data, aes(x = soma_cell_ratio, y = transformation_index, color = phenotype)) +
  geom_point(alpha = 0.5, size = 2) +
  scale_color_manual(values = pheno_colors) +
  labs(title = "Morphological Space",
       x = "Soma:Cell Ratio", y = "Transformation Index",
       color = "Phenotype") +
  theme_pub +
  # Draw threshold lines
  geom_vline(xintercept = thresholds$amoeboid_soma_ratio, linetype = "dashed", alpha = 0.3) +
  geom_vline(xintercept = thresholds$bushy_soma_ratio, linetype = "dashed", alpha = 0.3) +
  geom_vline(xintercept = thresholds$hypertrophic_soma_ratio, linetype = "dashed", alpha = 0.3) +
  geom_hline(yintercept = thresholds$hypertrophic_ti_max, linetype = "dashed", alpha = 0.3)
ggsave("plot_morphological_space.png", p_scatter, width = 9, height = 7, dpi = 300)

# --- 5. Box plots of key metrics by phenotype ---
p_scr <- ggplot(data, aes(x = phenotype, y = soma_cell_ratio, fill = phenotype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, size = 0.8, alpha = 0.3) +
  scale_fill_manual(values = pheno_colors) +
  labs(title = "Soma:Cell Ratio by Phenotype", y = "Soma:Cell Ratio", x = "") +
  theme_pub + theme(legend.position = "none")

p_ti <- ggplot(data, aes(x = phenotype, y = transformation_index, fill = phenotype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, size = 0.8, alpha = 0.3) +
  scale_fill_manual(values = pheno_colors) +
  labs(title = "Transformation Index by Phenotype", y = "Transformation Index", x = "") +
  theme_pub + theme(legend.position = "none")

p_round <- ggplot(data, aes(x = phenotype, y = roundness, fill = phenotype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, size = 0.8, alpha = 0.3) +
  scale_fill_manual(values = pheno_colors) +
  labs(title = "Roundness by Phenotype", y = "Roundness", x = "") +
  theme_pub + theme(legend.position = "none")

p_area <- ggplot(data, aes(x = phenotype, y = mask_area, fill = phenotype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.2, size = 0.8, alpha = 0.3) +
  scale_fill_manual(values = pheno_colors) +
  labs(title = "Cell Area by Phenotype", y = "Cell Area (µm²)", x = "") +
  theme_pub + theme(legend.position = "none")

combined_metrics <- ggarrange(p_scr, p_ti, p_round, p_area,
                               ncol = 2, nrow = 2, common.legend = FALSE)
ggsave("plot_phenotype_metrics.png", combined_metrics, width = 12, height = 10, dpi = 300)

# --- 6. Per-image phenotype heatmap ---
if (has_treatment) {
  heatmap_data <- data %>%
    group_by(image_name, treatment, phenotype) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(image_name, treatment) %>%
    mutate(percent = n / sum(n) * 100) %>%
    ungroup()

  p_heatmap <- ggplot(heatmap_data, aes(x = phenotype, y = image_name, fill = percent)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(percent, 0)), size = 3) +
    scale_fill_gradient(low = "white", high = "#2196F3", name = "% Cells") +
    facet_grid(treatment ~ ., scales = "free_y", space = "free_y") +
    labs(title = "Phenotype Distribution per Image", x = "", y = "") +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.y = element_text(size = 8),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      strip.text = element_text(face = "bold")
    )
  ggsave("plot_phenotype_heatmap.png", p_heatmap, width = 10,
         height = max(6, length(unique(data$image_name)) * 0.4), dpi = 300)
}

# ============================================================================
# DONE
# ============================================================================

cat("\n============================================\n")
cat("All outputs saved:\n")
cat("  ", output_path, " (data + phenotype column)\n")
cat("  ", summary_path, " (per-image counts + percentages)\n")
cat("  plot_phenotype_pie.png\n")
cat("  plot_phenotype_bar.png\n")
cat("  plot_morphological_space.png\n")
cat("  plot_phenotype_metrics.png (4-panel box plots)\n")
if (has_treatment) {
  cat("  plot_phenotype_by_treatment.png\n")
  cat("  plot_phenotype_stacked.png\n")
  cat("  plot_phenotype_heatmap.png\n")
}
cat("============================================\n")
