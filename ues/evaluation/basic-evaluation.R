# basic-evaluation.R
# for UES/transformed query workloads

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(stringr)
library(ggplot2)
library(viridis)
library(scales)

# Data loading ---- 

df_orig <- read_csv("workloads/job-ues-results-orig-nofk.csv") %>%
  select(-flattened_query) %>%
  rename(rt_total = query_rt_total) %>%
  mutate(foreign_keys = FALSE, workload = "ues")

# use the run with median runtime as representative for this experiment
# in order to compute a meaningful median, we first throw away the longest run
# to obtain an odd number of experiment runs
repr_orig <- df_orig %>%
  group_by(run) %>%
  summarise(rt_total = sum(rt_total)) %>%
  filter(rt_total < max(rt_total)) %>%
  filter(rt_total == median(rt_total)) %>%
  pull(run)


df_flattened <- read_csv("workloads/job-ues-results-flattened-nofk.csv") %>%
  select(-query) %>%
  rename(query = flattened_query,
         query_result = flattened_query_result,
         rt_total = flattened_query_rt_total) %>%
  mutate(foreign_keys = FALSE, workload = "flattened")

# use the run with median runtime as representative for this experiment
repr_flat <- df_flattened %>%
  group_by(run) %>%
  summarise(rt_total = sum(rt_total)) %>%
  filter(rt_total < max(rt_total)) %>%
  filter(rt_total == median(rt_total)) %>%
  pull(run)

df_orig_fk <- read_csv("workloads/job-ues-results-orig-fks.csv") %>%
  select(-flattened_query) %>%
  rename(rt_total = query_rt_total) %>%
  mutate(foreign_keys = TRUE, workload = "ues")

# use the run with median runtime as representative for this experiment
repr_orig_fk <- df_orig_fk %>%
  group_by(run) %>%
  summarise(rt_total = sum(rt_total)) %>%
  filter(rt_total < max(rt_total)) %>%
  filter(rt_total == median(rt_total)) %>%
  pull(run)

df_flattened_fk <- read_csv("workloads/job-ues-results-flattened-fks.csv") %>%
  select(-query) %>%
  rename(query = flattened_query,
         query_result = flattened_query_result,
         rt_total = flattened_query_rt_total)  %>%
  mutate(foreign_keys = TRUE, workload = "flattened")

# use the run with median runtime as representative for this experiment
repr_flat_fk <- df_flattened_fk %>%
  group_by(run) %>%
  summarise(rt_total = sum(rt_total)) %>%
  filter(rt_total < max(rt_total)) %>%
  filter(rt_total == median(rt_total)) %>%
  pull(run)

df_compl <- bind_rows(df_orig, df_flattened, df_orig_fk, df_flattened_fk)
df_repr <- bind_rows(
  df_orig %>% filter(run == repr_orig),
  df_flattened %>% filter(run == repr_flat),
  df_orig_fk %>% filter(run == repr_orig_fk),
  df_flattened_fk %>% filter(run == repr_flat_fk))

df_rt_dist <- df_compl %>%
  group_by(workload, foreign_keys, run) %>%
  summarise(runtime = sum(rt_total)) %>%
  mutate(experiment = paste(workload, ifelse(foreign_keys, "FKs", "no FKs"), sep = " :: "))

# Experiment overview ----
df_agg <- df_repr %>%
  group_by(workload, foreign_keys) %>%
  summarise(max_rt = max(rt_total),
            rt_total = sum(rt_total)) %>%
  mutate(top_heaviness = max_rt / rt_total,
         rt_remain = rt_total - max_rt,
         experiment = paste(workload, ifelse(foreign_keys, "FKs", "no FKs"), sep = " :: ")) %>%
  arrange(foreign_keys, workload)
ggplot(df_agg, aes(x = experiment, y = rt_total, fill = experiment)) +
  geom_col() +
  labs(title = "Total runtime of the different experiments",
       x = "Setting", y = "Runtime [seconds]",
       fill = "Setting") +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  theme_bw()

# Outlier analysis ----

# the longest running query
q_max <- df_repr %>% filter(rt_total == max(rt_total))

# number of outliers per workload
df_domin <- df_repr %>%
  filter(rt_total >= quantile(rt_total, 0.95)) %>%
  select(label, foreign_keys, workload, rt_total) %>%
  arrange(desc(rt_total)) %>%
  mutate(experiment = paste(workload, ifelse(foreign_keys, "FKs", "no FKs"), sep = " :: "))

ggplot(df_domin, aes(x = experiment, fill = experiment)) +
  geom_histogram(stat = "count") +
  labs(title = "Queries with exceptional long runtime",
       subtitle = "Included are all queries with a runtime above the 95% quantile",
       x = "Setting", y = "Number of queries",
       fill = "Setting",
       caption = "Using Foreign keys makes the optimizer more susceptible to very bad query plans.") +
  scale_y_continuous(breaks = pretty_breaks()) +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  theme_bw()

# duration of the longest queries per workload
ggplot(df_agg %>%
         pivot_longer(cols = c(max_rt, rt_remain),
                      names_to = "fraction",
                      values_to = "runtime"),
       aes(x = experiment, y = runtime, fill = fraction)) +
  geom_col() +
  labs(title = "Share of the longest running query on the total runtime",
       x = "Setting", y = "Runtime [seconds]",
       fill = "Runtime") +
  scale_fill_viridis(labels = c("Longest query", "Remainder"),
                     option = "cividis", discrete = TRUE, direction = -1) +
  theme_bw()


# Analysis within workload ----
df_nofk <- df_repr %>%
  filter(foreign_keys == FALSE) %>%
  select(-run)
df_nofk <- inner_join(
  df_nofk %>% filter(workload == "ues") %>%
    rename(orig_query = query, orig_result = query_result,  orig_rt = rt_total) %>%
    select(-foreign_keys, -workload),
  df_nofk %>% filter(workload == "flattened") %>%
    rename(trans_query = query, trans_result = query_result, trans_rt = rt_total) %>%
    select(-foreign_keys, -workload),
  by = "label"
)
df_nofk$rt_change_abs <- df_nofk$orig_rt - df_nofk$trans_rt
df_nofk$rt_change_pct <- df_nofk$trans_rt / df_nofk$orig_rt
df_nofk$n_joins <- str_count(df_nofk$trans_query, regex("join", ignore_case = TRUE))
df_nofk$base_table <- str_match(df_nofk$trans_query, regex("from (?<basetable>\\w+)", ignore_case = TRUE))[,"basetable"]
df_nofk$change_type <- ifelse(df_nofk$rt_change_pct < 1, "speedup", "slowdown")
#str_match_all(df_nofk$trans_query, regex("join (?<jointable>\\w+)", ignore_case = TRUE))
df_nofk$card <- as.numeric(sapply(str_match_all(df_nofk$trans_result, "'Actual Rows': (\\d+),"),
                                  function(res) res[2,2]))
df_nofk <- df_nofk %>% arrange(rt_change_pct)

df_fast <- df_nofk %>%
  filter(rt_change_pct <= quantile(rt_change_pct, 0.15)) %>%
  arrange(rt_change_pct)
df_slow <- df_nofk %>%
  filter(rt_change_pct >= quantile(rt_change_pct, 0.85)) %>%
  arrange(desc(rt_change_pct))

# Overview: performance distribution
ggplot(df_nofk, aes(x = 1:nrow(df_nofk), y = rt_change_pct, color = change_type, fill = change_type)) +
  geom_line() +
  geom_hline(aes(yintercept = 1, linetype = "Normal runtime"), size = .2) +
  geom_vline(aes(xintercept = nrow(df_nofk) / 2, linetype = "50% workload"), size = .2) +
  ylim(0, NA) +
  scale_y_log10() +
  labs(x = "Query (ordered by runtime change factor)", y = "Runtime change factor",
       fill = "Runtime change", color = "Runtime change", linetype = "Marker") +
  scale_fill_viridis(discrete = TRUE) +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(axis.text.x = element_blank())

# Correlation w/ number of joins?
ggplot(df_nofk, aes(x = n_joins, y = rt_change_pct, color = trans_rt)) +
  geom_point() +
  scale_y_log10() +
  labs(x = "Number of joins", y = "Runtime change factor",
       color = "Transformed RT") +
  scale_color_viridis() +
  theme_bw()

# Correlation w/ base table?
ggplot(df_nofk, aes(x = base_table, y = rt_change_pct, color = trans_rt)) +
  geom_point() +
  scale_y_log10() +
  labs(x = "Base table", y = "Runtime change factor",
       color = "Transformed RT") +
  scale_color_viridis() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation w/ cardinality?
ggplot(df_nofk, aes(x = card, y = rt_change_pct, color = trans_rt)) +
  geom_point() +
  scale_y_log10() +
  labs(x = "Result cardinality", y = "Runtime change factor",
       color = "Transformed RT") +
  scale_color_viridis() +
  theme_bw()

# df_plt <- bind_rows(df_fk %>% mutate(foreign_keys = TRUE),
#                     df_nofk %>% mutate(foreign_keys = FALSE)) %>%
#   mutate(foreign_keys = ifelse(foreign_keys, "on", "off"))
# ggplot(df_plt, aes(x = foreign_keys, y = rt_change, fill = foreign_keys)) +
#   geom_boxplot() +
#   scale_y_log10() +
#   labs(title = "Abs. difference in runtime original -> transformed queries",
#        x = "Foreign keys", y = "Runtime difference",
#        fill = "Foreign keys") +
#   scale_fill_viridis(option = "cividis", discrete = TRUE) +
#   theme_bw()
