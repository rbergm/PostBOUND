# basic-evaluation.R
# for UES/transformed query workloads

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(ggplot2)
library(viridis)
library(scales)

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

ggplot(df_rt_dist, aes(x = experiment, y = runtime, fill = experiment)) +
  geom_boxplot()

# the longest running query
q_max <- df_repr %>% filter(rt_total == max(rt_total))

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

df_agg <- df_repr %>%
  group_by(workload, foreign_keys) %>%
  summarise(max_rt = max(rt_total),
            rt_total = sum(rt_total)) %>%
  mutate(top_heaviness = max_rt / rt_total,
         experiment = paste(workload, ifelse(foreign_keys, "FKs", "no FKs"), sep = " :: "))
ggplot(df_agg, aes(x = experiment, y = rt_total, fill = experiment)) +
  geom_bar(stat = "identity") +
  labs(title = "Total runtime of the different experiments",
       x = "Setting", y = "Runtime",
       fill = "Setting") +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  theme_bw()

ggplot(df_agg %>%
         pivot_longer(cols = c(max_rt, rt_total),
                      names_to = "fraction",
                      values_to = "runtime"),
       aes(x = experiment, y = runtime, fill = fraction)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Share of the longest running query on the total runtime",
       x = "Setting", y = "Runtime",
       fill = "Measure") +
  scale_fill_viridis(labels = c("Longest query", "Total runtime"),
                     option = "cividis", discrete = TRUE, direction = -1) +
  theme_bw()
