# query-selection-eval.R
# Comparse two different query selection strategies from the workload
# repetitions. The first strategy simply selects all queries from the workload
# with minimum total runtime out of all repetitions.
# The second strategy operates on a per-query basis and selects among each
# repetition of the query the one with minimum runtime. Therfore, the resulting
# data set will span multiple repetitions in the general case

library(readr)
library(dplyr)
library(tidyr)

library(ggplot2)
library(viridis)
library(scales)

df_workload <- read_csv("workloads/job-ues-eval-idxnlj.csv")
df_queries <- read_csv("workloads/job-ues-eval-idxnlj-queryopt.csv")

df <- full_join(df_workload %>% select(label, rt_total_hinted),
                df_queries %>% select(label, rt_total_hinted),
                suffix = c("_wopt", "_qopt"),
                by = "label") %>%
  mutate(qopt_speedup = rt_total_hinted_wopt - rt_total_hinted_qopt) %>%
  arrange(desc(qopt_speedup))

ggplot(df, aes(x = 1:nrow(df), y = qopt_speedup, fill = qopt_speedup >= 0)) +
  geom_line() +
  geom_area() +
  labs(title = "Runtimed improvement via per-query selection",
       subtitle = paste("Per-query selection chooses the fastest query run, whereas",
                        "the default selection chooses the overall fastest workload",
                        sep = "\n"),
       x = "JOB query (ordered by selection runtime improvement)", y = "Selection runtime improvement [seconds]") +
  theme_bw() +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  theme(legend.position = "none", axis.text.x = element_blank())

