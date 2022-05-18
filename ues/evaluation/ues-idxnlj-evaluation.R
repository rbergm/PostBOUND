# ues-idxnlj-evaluation.R
# Analysis of the impact of query hints on the runtime

# Setup ====
library(readr)
library(dplyr)
library(tidyr)

library(ggplot2)
library(viridis)

# Data loading ====
df_full <- read_csv("workloads/job-ues-results-idxnlj.csv")

# Select the fastest runs as representatives for each workload
best_runs <- df_full %>%
  group_by(workload, run) %>%
  summarise(rt_total = sum(rt_total)) %>%
  group_by(workload) %>%
  filter(rt_total == min(rt_total))
df_repr <- semi_join(df_full, best_runs, by = c("workload", "run"))

# Select only queries with actual hints for further comparisons
hinted_queries <- df_repr %>% filter(workload == "ues_idxnlj", !is.na(query_hint))
orig_queries <- df_repr %>% filter(workload == "ues") %>% semi_join(hinted_queries, by = "label")
df <- full_join(
  orig_queries %>% select(label, rt_total) %>% rename(rt_orig = rt_total),
  hinted_queries %>% select(label, rt_total) %>% rename(rt_hinted = rt_total),
  by = "label"
) %>% mutate(runtime_diff = rt_orig - rt_hinted)

# Workload overview ====

# Per-query runtime differences
df_plt <- df %>%
  mutate(faster_setting = ifelse(runtime_diff > 0, "UES w/ IdxNLJ", "Pure UES")) %>%
  arrange(desc(runtime_diff))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = runtime_diff,
               color = faster_setting, fill = faster_setting)) +
  geom_line() +
  geom_area() +
  labs(title = "Runtime difference between pure UES queries and hinted queries",
       subtitle = "Hints force subqueries to be executed as IdxNLJ",
       x = "JOB query (ordered by IdxNLJ speedup)", y = "IdxNLJ speedup [seconds]",
       color = "Faster setting", fill = "Faster setting") +
  theme_bw() +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  scale_color_viridis(option = "inferno", discrete = TRUE) +
  theme(axis.text.x = element_blank())

