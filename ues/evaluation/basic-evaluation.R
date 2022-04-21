# basic-evaluation.R
# for UES/transformed query workloads

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(ggplot2)

df_orig <- read_csv("workloads/job-ues-results-orig-nofk.csv") %>%
  select(-flattened_query) %>%
  rename(rt_total = query_rt_total) %>%
  mutate(foreign_keys = FALSE, workload = "ues")

df_flattened <- read_csv("workloads/job-ues-results-flattened-nofk.csv") %>%
  select(-query) %>%
  rename(query = flattened_query,
         query_result = flattened_query_result,
         rt_total = flattened_query_rt_total) %>%
  mutate(foreign_keys = FALSE, workload = "flattened")

df_orig_fk <- read_csv("workloads/job-ues-results-orig-fks.csv") %>%
  select(-flattened_query) %>%
  rename(rt_total = query_rt_total) %>%
  mutate(foreign_keys = TRUE, workload = "ues")

df_flattened_fk <- read_csv("workloads/job-ues-results-flattened-fks.csv") %>%
  select(-query) %>%
  rename(query = flattened_query,
         query_result = flattened_query_result,
         rt_total = flattened_query_rt_total)  %>%
  mutate(foreign_keys = TRUE, workload = "flattened")

df <- bind_rows(df_orig, df_flattened, df_orig_fk, df_flattened_fk)

df_rt_dist <- df %>%
  group_by(workload, foreign_keys, run) %>%
  summarise(runtime = sum(rt_total)) %>%
  mutate(experiment = paste(workload, ifelse(foreign_keys, "FKs", "no FKs"), sep = " :: "))

ggplot(df_rt_dist, aes(x = experiment, y = runtime, fill = experiment)) +
  geom_boxplot()
