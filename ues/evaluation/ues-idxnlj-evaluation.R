# ues-idxnlj-evaluation.R
# Analysis of the impact of query hints on the runtime

# Setup ====
library(readr)
library(dplyr)
library(tidyr)
library(stringr)

library(ggplot2)
library(viridis)
library(scales)

# Data loading ====
df_full <- read_csv("workloads/job-ues-results-idxnlj.csv")

# # Select the fastest runs as representatives for each workload
# best_runs <- df_full %>%
#   group_by(workload, run) %>%
#   summarise(rt_total = sum(rt_total)) %>%
#   group_by(workload) %>%
#   filter(rt_total == min(rt_total))
# df_repr <- semi_join(df_full, best_runs, by = c("workload", "run"))
# 
# # Select only queries with actual hints for further comparisons
# hinted_queries <- df_repr %>% filter(workload == "ues_idxnlj", !is.na(query_hint))
# orig_queries <- df_repr %>% filter(workload == "ues") %>% semi_join(hinted_queries, by = "label")
# df_queries <- full_join(
#   orig_queries %>% select(label, rt_total, query) %>% rename(rt_orig = rt_total),
#   hinted_queries %>% select(label, rt_total) %>% rename(rt_hinted = rt_total),
#   by = "label"
# ) %>% mutate(runtime_diff = rt_orig - rt_hinted,
#              n_subqueries = str_count(query, regex("select ", ignore_case = TRUE)) - 1)

# Alternative selection
df_repr <- df_full %>%
  filter(workload == "ues" | workload == "ues_idxnlj") %>%
  group_by(workload, label) %>%
  arrange(rt_total) %>%
  slice_head() %>%
  ungroup()
df_queries <- inner_join(
  df_repr %>% filter(workload == "ues") %>% select(label, rt_total, query) %>% rename(rt_orig = rt_total),
  df_repr %>% filter(workload == "ues_idxnlj", !is.na(query_hint)) %>% select(label, rt_total) %>% rename(rt_hinted = rt_total),
  by = "label"
) %>% mutate(runtime_diff = rt_orig - rt_hinted,
             n_subqueries = str_count(query, regex("select ", ignore_case = TRUE)) - 1)

df_features <- read_csv("workloads/job-ues-eval-idxnlj-queryopt.csv")
df_features <- full_join(df_features,
                         df_features %>%
                           group_by(label) %>%
                           tally(name = "n_subqueries"),
                         on = "label")

# Workload overview ====

# Per-query runtime differences
df_plt <- df_queries %>%
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

# Runtime differences, depending on the number of subqueries
df_plt <- df_queries %>% mutate(n_subqueries = as.factor(n_subqueries),
                                faster_setting = ifelse(runtime_diff > 0, "UES w/ IdxNLJ", "Pure UES"))
ggplot(df_plt, aes(x = n_subqueries, y = runtime_diff, color = faster_setting)) +
  geom_point() +
  labs(title = "Correlation between the number of subqueries and speedup per query",
       x = "Number of subqueries",  y = "IdxNLJ speedup [seconds]",
       color = "Faster setting") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# Subquery analysis ====
df_plt <- df_features %>% mutate(n_subqueries = as.factor(n_subqueries))

# Runtime differences depending on the number of outgoing tuples
ggplot(df_plt, aes(x = subquery_rows, y = runtime_diff, color = n_subqueries)) +
  geom_point() + 
  labs(x = "Number of tuples emitted from the subquery", y = "IdxNLJ speedup [seconds]",
       color = "Number of subqueries\nin the query") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# Runtime differences depending on the number of FK tuples
ggplot(df_plt, aes(x = subquery_fk_rows, y = runtime_diff, color = n_subqueries)) +
  geom_point() + 
  labs(x = "Number of tuples in the FK table", y = "IdxNLJ speedup [seconds]",
       color = "Number of subqueries\nin the query") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# Runtime differences depending on the number of PK tuples
ggplot(df_plt, aes(x = subquery_pk_rows, y = runtime_diff, color = n_subqueries)) +
  geom_point() + 
  labs(x = "Number of tuples in the PK table", y = "IdxNLJ speedup [seconds]",
       color = "Number of subqueries\nin the query") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)

# Only consider each query once and use the maximum feature value per subquery
df_plt <- df_features %>%
  group_by(label) %>%
  summarise(subquery_pk_rows = max(subquery_pk_rows),
            subquery_fk_rows = max(subquery_fk_rows),
            subquery_rows = max(subquery_rows),
            runtime_diff = unique(runtime_diff),
            n_subqueries = unique(n_subqueries)) %>%
  mutate(n_subqueries = as.factor(n_subqueries))
ggplot(df_plt, aes(x = subquery_pk_rows, y = runtime_diff, color = n_subqueries)) +
  geom_point() + 
  labs(x = "Number of tuples in the PK table", y = "IdxNLJ speedup [seconds]",
       color = "Number of subqueries\nin the query") +
  theme_bw() +
  scale_color_viridis(option = "cividis", discrete = TRUE, direction = -1)


# Queries with exactly 2 subqueries ====
df_2sqs <- df_features %>%
  semi_join(df_queries %>% filter(n_subqueries == 2), by = "label") %>%
  group_by(label) %>%
  summarise(runtime_diff = unique(runtime_diff),
           rows_max = max(subquery_rows),
           rows_min = min(subquery_rows),
           pk_rows_max = max(subquery_pk_rows),
           pk_rows_min = min(subquery_pk_rows),
           fk_rows_max = max(subquery_fk_rows),
           fk_rows_min = min(subquery_fk_rows))

# Influence of subquery cardinality
ggplot(df_2sqs, aes(x = rows_min, y = rows_max, color = runtime_diff)) +
  geom_point() +
  labs(title = "Influence of the total subquery cardinality",
       subtitle = paste("Subquery cardinality = total number of emitted tuples",
                        "Shown are only queries with exactly two subqueries",
                        "Each axis corresponds to one subquery",
                        sep = "\n"),
       x = "Minimum cardinality", y = "Maximum cardinality",
       color = "Hint speedup") +
  theme_bw() +
  scale_color_viridis(option = "cividis")

# Influence of PK cardinality
ggplot(df_2sqs, aes(x = pk_rows_min, y = pk_rows_max, color = runtime_diff)) +
  geom_point() +
  labs(title = "Influence of the primary key cardinality",
       subtitle = paste("Shown are only queries with exactly two subqueries",
                        "Each axis corresponds to one subquery",
                        sep = "\n"),
       x = "Minimum cardinality", y = "Maximum cardinality",
       color = "Hint speedup") +
  theme_bw() +
  scale_color_viridis(option = "cividis")

# Influence of FK cardinality
ggplot(df_2sqs, aes(x = fk_rows_min, y = fk_rows_max, color = runtime_diff)) +
  geom_point() +
  labs(title = "Influence of the foreign key cardinality",
       subtitle = paste("Shown are only queries with exactly two subqueries",
                        "Each axis corresponds to one subquery",
                        sep = "\n"),
       x = "Minimum cardinality", y = "Maximum cardinality",
       color = "Hint speedup") +
  theme_bw() +
  scale_color_viridis(option = "cividis")


# Queries with just one subquery ====
df_1sq <- df_features %>%
  semi_join(df_queries %>% filter(n_subqueries == 1), by = "label")

# Influence of PK/FK counts on total speedup
ggplot(df_1sq, aes(x = subquery_pk_rows, y = subquery_fk_rows, color = runtime_diff)) +
  geom_point() +
  scale_x_continuous(trans = log1p_trans(), breaks = trans_breaks("log1p", "expm1"), labels = label_scientific()) +
  #scale_x_log10() +
  scale_y_continuous(trans = log1p_trans(), breaks = trans_breaks("log1p", "expm1"), labels = label_scientific()) +
  #scale_y_log10() +
  labs(title = "Influence of the primary key / foreign key cardinality",
       subtitle = paste("Shown are only queries with just one subquery",
                        sep = "\n"),
       x = "Primary key cardinality", y = "Foreign key cardinality",
       color = "Hint speedup") +
  theme_bw() +
  scale_color_viridis(option = "cividis")

df_plt <- df_1sq %>%
  filter(subquery_pk_rows > 0, subquery_fk_rows > 0) %>%
  mutate(faster_setting = ifelse(runtime_diff > 0, "UES w/ IdxNLJ", "Pure UES")) %>%
  arrange(desc(runtime_diff))
ggplot(df_plt, aes(x = subquery_pk_rows, y = runtime_diff)) + geom_point()

ggplot(df_plt, aes(x = subquery_pk_rows, y = subquery_fk_rows, color = runtime_diff)) +
  geom_point() +
  geom_text(aes(label = label), nudge_y = 0.3, check_overlap = TRUE) +
  scale_x_log10(breaks = 10^(1:7)) +
  scale_y_log10(breaks = 10^(1:7)) +
  labs(title = "Influence of the primary key / foreign key cardinality",
       subtitle = paste("Shown are only queries with just one subquery",
                        "0 FK/PK situations are removed",
                        sep = "\n"),
       x = "Primary key cardinality", y = "Foreign key cardinality",
       color = "Hint speedup") +
  theme_bw() +
  scale_color_viridis(option = "viridis")

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
