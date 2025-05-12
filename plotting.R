library(tidyverse)
library(tikzDevice)
library(RColorBrewer)

print_plot <- function(plot, name, width=3.5, height=2.5){
  tex_name <- sprintf("results/fig/%s.tex", name)
  png_name <- sprintf("results/fig/%s.png", name)
  tex_width <- width; tex_height <- height
  png_width <- tex_width * 4; png_height <- tex_height * 4
  
  # Define LaTeX preamble to use sans-serif font
  sans_preamble <- c(
    "\\usepackage{pgfplots}",
    "\\pgfplotsset{compat=newest}",
    "\\usepackage[utf8]{inputenc}",
    "\\usepackage[T1]{fontenc}",
    "\\usepackage{sfmath}",  # Use sfmath for sans-serif math
    "\\renewcommand{\\familydefault}{\\sfdefault}"  # Set default font to sans-serif
  )
  
  # Use tikzDevice with custom preamble
  tikz(file = tex_name, width = tex_width, height = tex_height, sanitize = TRUE,
       documentDeclaration = "\\documentclass[12pt]{standalone}",
       packages = sans_preamble)
  print(plot)
  dev.off()
  
  # PNG export remains unchanged
  png(file = png_name, width = png_width, height = png_height, units = "cm", res = 150)
  print(plot)
  dev.off()
}

# First plot: MRR data
mrr_df <- read_tsv("results/experiment_mrrs.tsv", col_names = FALSE)

colnames(mrr_df) <- c(
  "user_id", "local_mrr",
  paste0("naive_mrr_", 0:10),
  paste0("defense_mrr_", 0:10),
  paste0("oracle_mrr_", 0:10)
)

x_pct <- seq(0, 100, 10)
naive_mrr <- map_dbl(0:10, ~ mean(mrr_df[[paste0("naive_mrr_", .x)]]))
defense_mrr <- map_dbl(0:10, ~ mean(mrr_df[[paste0("defense_mrr_", .x)]]))
oracle_mrr <- map_dbl(0:10, ~ mean(mrr_df[[paste0("oracle_mrr_", .x)]]))
mean_loc <- mean(mrr_df$local_mrr)

oracle_mrr[1] <- naive_mrr[1]
oracle_mrr[11] <- mean_loc

df <- tibble(
  pct = x_pct,
  Naive = naive_mrr,
  `With Defense` = defense_mrr,
  Oracle = oracle_mrr
) %>%
  pivot_longer(-pct, names_to = "Metric", values_to = "Value")

h_line_df <- tibble(
  pct = range(df$pct),
  Value = mean_loc,
  Metric = "Local-only"
)

random_line_df <- tibble(
  pct = range(df$pct),
  Value = 1/5.5,
  Metric = "Random"
)

p1 <- ggplot() +
  geom_line(data = df, aes(pct, Value, color = Metric, linetype = Metric)) +
  geom_point(data = df, aes(pct, Value, shape = Metric, color = Metric)) +
  geom_line(data = h_line_df, aes(pct, Value), color = "black", linetype = "dotted", size = 1) +
  geom_line(data = random_line_df, aes(pct, Value), color = "grey50", linetype = "dotdash", size = 1) +
  geom_text(data = tibble(pct = 0, Value = mean_loc, label = "Local-only"), 
            aes(pct, Value, label = label), vjust = -0.5, hjust = 0, size = 3, color = "black") +
  geom_text(data = tibble(pct = 0, Value = 1/5.5, label = "Random"), 
            aes(pct, Value, label = label), vjust = -0.5, hjust = 0, size = 3, color = "grey50") +
  scale_y_continuous(name = "MRR") +
  scale_color_manual(
    values = c(
      "With Defense" = "#368544",
      "Naive" = "#CC3329",
      "Oracle" = "#1E90FF"
    ),
    breaks = c("Naive", "With Defense", "Oracle")
  ) +
  scale_linetype_manual(
    values = c(
      "With Defense" = "twodash",
      "Naive" = "dashed",
      "Oracle" = "solid"
    ),
    breaks = c("Naive", "With Defense", "Oracle")
  ) +
  scale_shape_manual(
    values = c(
      "With Defense" = 16,
      "Naive" = 15,
      "Oracle" = 17
    ),
    guide = "none"
  ) +
  guides(color = guide_legend(override.aes = list(shape = c(15, 16, 17)))) +
  labs(x = "Adversarial Nodes in Network (%)", color = NULL, linetype = NULL) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.major = element_line(linetype = "dashed"),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 9),
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 9)
  )

print_plot(p1, "poisoning")

# Second plot: Sybil scenarios
sybil_df <- read_csv("results/sybil_scenarios.csv")

sybil_tidy <- sybil_df %>%
  pivot_longer(-SybilFraction, names_to = "Metric", values_to = "Value")

p2 <- ggplot() +
  geom_line(data = sybil_tidy, aes(SybilFraction, Value, color = Metric, linetype = Metric)) +
  geom_point(data = sybil_tidy, aes(SybilFraction, Value, shape = Metric, color = Metric)) +
  scale_y_continuous(name = "Donations Captured by Sybils (%)") +
  scale_color_brewer(
    palette = "Set1",
    breaks = c("Base", "No smoothing", "Aggressive weighting", "Dense Sybils", "Random")
  ) +
  scale_linetype_manual(
    values = c(
      "Base" = "dashed",
      "No smoothing" = "twodash",
      "Aggressive weighting" = "solid",
      "Dense Sybils" = "dotted",
      "Random" = "dotdash"
    ),
    breaks = c("Base", "No smoothing", "Aggressive weighting", "Dense Sybils", "Random")
  ) +
  scale_shape_manual(
    values = c(
      "Base" = 15,
      "No smoothing" = 16,
      "Aggressive weighting" = 17,
      "Dense Sybils" = 18,
      "Random" = 19
    ),
    guide = "none"
  ) +
  guides(color = guide_legend(
    override.aes = list(shape = c(15, 16, 17, 18, 19)),
    keywidth = 1.5,
    keyheight = 0.5
  )) +
  labs(x = "Sybil Nodes in Network (%)", color = NULL, linetype = NULL) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = c(0.05, 0.95),
    legend.justification = c(0, 1),
    legend.margin = margin(0, 12, 4, 2),
    legend.background = element_rect(fill = "white", color = "black", size = 0.1),
    panel.grid.major = element_line(linetype = "dashed"),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 9),
    plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt")  # Further reduce padding
  )

print_plot(p2, "sybil_scenarios", height=2)

df_100 <- read_csv("results/chronological_100.csv") %>%
  mutate(docs = "100")
df_1000 <- read_csv("results/chronological_1000.csv") %>%
  mutate(docs = "1000")
df_5000 <- read_csv("results/chronological_5000.csv") %>%
  mutate(docs = "5000")

df_all <- bind_rows(df_100, df_1000, df_5000) %>%
  mutate(timestamp = ymd_hms(timestamp)) %>%
  group_by(docs) %>%
  mutate(hours = as.numeric(difftime(timestamp, min(timestamp), units = "hours"))) %>%
  ungroup() %>% 
  filter(hours <= 10)

# Plot
p3 <- ggplot(df_all, aes(x = hours, y = test_acc, color = docs, linetype = docs, shape = docs)) +
  geom_line() +
  geom_point(size = 1) +
  scale_color_brewer(
    palette = "Set1",
    labels = c("100" = "100 docs", "1000" = "1000 docs", "5000" = "5000 docs")
  ) +
  scale_linetype_manual(
    values = c("100" = "dashed", "1000" = "solid", "5000" = "dotdash"),
    labels = c("100" = "100 docs", "1000" = "1000 docs", "5000" = "5000 docs")
  ) +
  scale_shape_manual(
    values = c("100" = 15, "1000" = 16, "5000" = 17),
    labels = c("100" = "100 docs", "1000" = "1000 docs", "5000" = "5000 docs")
  ) +
  guides(
    color = guide_legend(override.aes = list(shape = c(15, 16, 17))),
    shape = "none"
  ) +
  labs(x = "Hours of Training",
       y = "Accuracy",
       color = NULL,
       linetype = NULL) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.background = element_blank(),     # no frame
    legend.box.background = element_blank(), # no surrounding box
    legend.key = element_blank(),            # no key boxes
    legend.margin = margin(0, 0, 0, 0),
    legend.box.spacing = unit(2, "pt"),
    panel.grid.major = element_line(linetype = "dashed"),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 9),
    plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt")
  )

print_plot(p3, "dsi", height=2)


# Display both plots
p1
p2
p3
