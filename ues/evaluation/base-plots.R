# base-plots.R

library(readr)
library(dplyr)
library(tidyr)
library(stringr)

library(gtools)

library(ggplot2)
library(viridis)
library(scales)

select_best_query_repetition <- function(workload) {
  selected <- workload %>%
    group_by(label) %>%
    summarise(min_rt = min(query_rt_total))

  return(selected %>% arrange(order(mixedorder(label))))
}

# Base results of JOB-Workload
job_base <- read_csv("workloads/job-ues-results-implicit.csv") %>% select_best_query_repetition()
ggplot(job_base, aes(x = label, y = min_rt)) +
  geom_col() +
  labs(x = "Query (ordered by label)", y = "Measured runtime [seconds]") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())
ggsave("plots/plot-job-implicit.pdf")

longest_base_queries <- job_base %>% slice_max(min_rt, n = 10)
print(paste("Longest query fraction", sum(longest_base_queries$min_rt) / sum(job_base$min_rt)))

# Unmodified UES results
job_ues <- read_csv("workloads/job-ues-results-rebuild.csv") %>% select_best_query_repetition()
ggplot(job_ues, aes(x = label, y = min_rt)) +
  geom_col() +
  labs(x = "Query (ordered by label)", y = "Measured runtime [seconds]") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())
ggsave("plots/plot-job-ues.pdf")

longest_ues_queries <- job_ues %>% slice_max(min_rt, n = 10)
print(paste("Longest query fraction", sum(longest_ues_queries$min_rt) / sum(job_ues$min_rt)))
