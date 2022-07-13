# ues-linearized-evaluation.R
# A more tailored analysis of the performance of UES subqueries

# Setup ====

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(tidyr)

library(ggplot2)
library(viridis)

# Data loading ====
df <- read_csv("workloads/job-ues-eval-linearized.csv")

# Workload overview ====

# Total runtime of each workload
df_plt <- df %>%
  group_by(label) %>%
  select(runtime_ues, runtime_ues_linear, ues_speedup_abs) %>%
  distinct() %>%  # queries with multiple subqueries will appear multiple times, remove those duplicates
  mutate(faster_query = ifelse(ues_speedup_abs > 0, "UES", "linearized UES")) %>%
  arrange(desc(ues_speedup_abs))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = ues_speedup_abs, fill = faster_query, color = faster_query)) +
  geom_line() +
  geom_area() +
  labs(title = "Runtime difference between pure UES and linearized UES queries",
       subtitle = "Shown is the absolute speedup of the pure UES query over its linearized counterpart.",
       x = "JOB query (ordered by speedup)", y = "Absolute speedup [seconds]",
       fill = "Faster query", color = "Faster query") +
  theme_bw() +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  scale_color_viridis(option = "inferno", discrete = TRUE) +
  theme(axis.text.x = element_blank())

# Subquery filter strength ====

# Correlation between filtered tuples and achieved speedup
df_plt <- df %>%
  mutate(subquery_impact = ifelse(ues_speedup_abs > 0, "speedup", "slowdown")) %>%
  arrange(desc(ues_speedup_abs))
ggplot(df_plt, aes(x = filter_strength_abs, y = ues_speedup_abs,
                   color = subquery_impact, shape = subquery_impact)) +
  geom_point() +
  geom_text(data = df_plt %>% head(n = 9), aes(label = label),
            nudge_y = 0.3, check_overlap = TRUE) +
  labs(title = "Correlation between filtered tuples and achieved speedup",
       subtitle = paste("Shown is the absolute speedup of the pure UES query over its linearized counterpart.",
                        "Filtered tuples only consider the foreign key table.",
                        "Some queries with outstanding speedup are labelled",
                        sep = "\n"),
       x = "Number of tuples removed in subquery", y = "Absolute speedup [seconds]",
       color = "Subquery impact", shape = "Subquery impact") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# Subquery duration ====

# Correlation between duration difference and achieved speedup
df_plt <- df %>%
  mutate(subquery_duration_diff = subquery_partner_runtime - subquery_runtime,
         subquery_impact = ifelse(ues_speedup_abs > 0, "speedup", "slowdown")) %>%
  arrange(desc(ues_speedup_abs))
ggplot(df_plt, aes(x = subquery_duration_diff, y = ues_speedup_abs,
                   color = subquery_impact, shape = subquery_impact)) +
  geom_point() +
  geom_text(data = df_plt %>% head(n = 9), aes(label = label),
            nudge_y = 0.3, check_overlap = TRUE) +
  labs(title = "Correlation between duration of the subquery branch and achieved speedup",
       subtitle = paste("Runtime difference is measured between join partner and subquery branch",
                        "Subquery impact is compared to linearized UES plan",
                        "Some queries with outstanding speedup are labelled",
                        sep = "\n"),
       x = "Branch runtime difference [seconds]", y = "Absolute speedup [seconds]",
       color = "Subquery impact", shape = "Subquery impact") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# Correlation between duration and achieved speedup
df_plt <- df %>%
  mutate(subquery_impact = ifelse(ues_speedup_abs > 0, "speedup", "slowdown")) %>%
  arrange(desc(ues_speedup_abs))
ggplot(df_plt, aes(x = subquery_runtime, y = ues_speedup_abs,
                   color = subquery_impact, shape = subquery_impact)) +
  geom_point() +
  geom_text(data = df_plt %>% head(n = 9), aes(label = label),
            nudge_y = 0.3, check_overlap = TRUE) +
  labs(title = "Correlation between duration of the subquery branch and achieved speedup",
       subtitle = paste("Subquery impact is compared to linearized UES plan",
                        "Some queries with outstanding speedup are labelled",
                        sep = "\n"),
       x = "Subquery runtime [seconds]", y = "Absolute speedup [seconds]",
       color = "Subquery impact", shape = "Subquery impact") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# FK table size ====

# Number of tuples in foreign key table of the subquery
df_plt <- df %>%
  mutate(subquery_impact = ifelse(ues_speedup_abs > 0, "speedup", "slowdown")) %>%
  arrange(desc(ues_speedup_abs))
ggplot(df_plt, aes(x = foreign_key_rows, y = ues_speedup_abs,
                   color = subquery_impact, shape = subquery_impact)) +
  geom_point() +
  geom_text(data = df_plt %>% head(n = 9), aes(label = label),
            nudge_y = 0.3, check_overlap = TRUE) +
  labs(title = "Correlation between size of the foreign key table and achieved speedup",
       subtitle = paste("Size is measured after filters",
                        "Some queries with outstanding speedup are labelled",
                        sep = "\n"),
       x = "Number of rows in foreign key table", y = "Absolute speedup [seconds]",
       color = "Subquery impact", shape = "Subquery impact") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)
