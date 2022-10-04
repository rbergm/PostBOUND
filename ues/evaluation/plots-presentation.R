## plots.presentation.R
# Generates the plots used for our BTW'23 paper
#

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(forcats)
library(tidyr)
library(stringr)

library(ggplot2)
library(viridis)
library(scales, warn.conflicts = FALSE)

df_cautious <- read_csv("evaluation/job-ues-eval-topk-exhaustive.csv")
df_cautious$setting <- factor(df_cautious$setting,
                          levels = c("UES", str_c("Top-", 1:5)),
                          ordered = TRUE)
df_approx <- read_csv("evaluation/job-ues-eval-topk-approx.csv")
df_approx$setting <- factor(df_approx$setting,
                            levels = c("UES", str_c("Top-", c(1, 5, 10, 20, 50, 100, 500))),
                            ordered = TRUE)


# - - - - - - - - - - - - - - - -
# 01: Runtime of the cautious algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_cautious %>% group_by(setting) %>% summarise(total_runtime = sum(execution_time))
ggplot(df_plt, aes(x = setting, y = total_runtime)) +
  geom_col() +
  labs(x = "Workload", y = "Total runtime [seconds]") +
  theme_bw()
ggsave("evaluation/plot-job-runtimes-topk-cautious.pdf")

# calculate the total difference in runtime between best/worst setting
grp_cautious <- df_cautious %>% group_by(setting) %>% summarise(total_rt = sum(execution_time))
max(grp_cautious$total_rt) - min(grp_cautious$total_rt)


# - - - - - - - - - - - - - - - -
# 02: Mean upper bounds of the cautious algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_cautious %>% group_by(setting) %>% summarise(mean_bound = mean(upper_bound))
ggplot(df_plt, aes(x = setting, y = mean_bound, group = 1)) +
  geom_line(linetype = "dotted") +
  geom_point() +
  labs(x = "Workload", y = "Mean upper bound") +
  theme_bw()
ggsave("evaluation/plot-job-mean-bounds-topk-cautious.pdf")

# calculate the ratio between mean upper bounds
grp_cautious <- df_cautious %>% group_by(setting) %>% summarise(mean_bound = mean(upper_bound))
min(grp_cautious$mean_bound) / max(grp_cautious$mean_bound)

# determine all queries that have an updated path/structure
join_path_updates <- inner_join(
  df_cautious %>% rename(join_path = ues_join_path) %>% filter(setting == "UES") %>% select(label, join_path),
  df_cautious %>% rename(join_path = ues_join_path) %>% filter(mode == "top-k") %>% select(label, setting, join_path),
  by = "label", suffix = c("_ues", "_topk")) %>%
  mutate(update = join_path_ues != join_path_topk)
join_path_updates %>% filter(update)

# calculate the runtime difference between UES/Top-k variant for updated queries
cautious_updated <- semi_join(df_cautious, join_path_updates %>% filter(update), by = c("label", "setting"))
update_runtimes <- inner_join(
  cautious_updated %>% select(label, setting, execution_time),
  df_cautious %>% filter(setting == "UES") %>% select(label, execution_time),
  by = "label", suffix = c("_ues", "_topk")) %>%
  mutate(runtime_diff = execution_time_ues - execution_time_topk)


# - - - - - - - - - - - - - - - -
# 03: Optimization time of the cautious algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_cautious %>% group_by(setting) %>% summarise(total_optimization_time = sum(optimization_time))
ggplot(df_plt, aes(x = setting, y = total_optimization_time, group = 1)) +
  geom_line(linetype = "dotted") +
  geom_point() +
  labs(x = "Workload", y = "Total optimization time [seconds]") +
  theme_bw()


# - - - - - - - - - - - - - - - -
# 11: runtime of the approximative algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_approx %>% group_by(setting, subquery_mode) %>% summarise(total_runtime = sum(execution_time))
ggplot(df_plt, aes(x = setting, y = total_runtime, fill = subquery_mode)) +
  geom_col(position = "dodge") +
  labs(x = "Workload", y = "Total runtime [seconds]",
       fill = "Subqueries") +
  scale_fill_viridis(option = "cividis", discrete = TRUE, end = 0.95) +
  theme_bw()
ggsave("evaluation/plot-job-runtimes-topk-approx.pdf")

# calculate the total difference in runtime between best/worst setting
grp_approx <- df_approx %>% group_by(setting, subquery_mode) %>% summarise(total_rt = sum(execution_time))
grp_approx %>% group_by(subquery_mode) %>% summarise(rt_diff = max(total_rt) - min(total_rt))


# - - - - - - - - - - - - - - - -
# 12: Median upper bounds of the approximative algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_approx %>%
  filter(subquery_mode == "linear") %>%
  group_by(setting) %>%
  summarise(median_bound = median(upper_bound), mean_bound = mean(upper_bound)) %>%
  pivot_longer(cols = c(median_bound, mean_bound), names_to = "aggregate")
ggplot(df_plt, aes(x = setting, y = value, color = aggregate, group = aggregate)) +
  geom_line(linetype = "dotted") +
  geom_point() +
  labs(x = "Workload",  y = "Median upper bound", color = "Aggregate") +
  scale_y_log10() +
  theme_bw()


# - - - - - - - - - - - - - - - -
# 13: Upper bound distribution of the approximative algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_approx %>% filter(subquery_mode == "linear")
ggplot(df_plt, aes(x = setting, y = upper_bound, fill = setting)) +
  geom_boxplot() +
  labs(x = "Workload", y = "Upper bounds") +
  scale_y_log10() +
  scale_fill_viridis(option = "cividis", discrete = TRUE, guide = "none", begin = 0.25) +
  theme_bw()
ggsave("evaluation/plot-job-box-bounds-topk-approx.pdf")

# calculate the ratio between mean upper bounds
grp_approx <- df_approx %>% filter(subquery_mode == "linear") %>% group_by(setting) %>% summarise(mean_bound = mean(upper_bound))
max(grp_approx$mean_bound) / min(grp_approx$mean_bound)

# determine all queries that have an updated path/structure
df_top50 <- df_approx %>% rename(join_path = ues_join_path) %>% filter(setting == "Top-50", subquery_mode == "smart", true_card > 0)
df_ues <- df_approx %>% rename(join_path = ues_join_path) %>% filter(setting == "UES", subquery_mode == "smart", true_card > 0)
join_path_updates <- inner_join(
  df_ues %>% select(label, join_path),
  df_top50 %>% select(label, setting, join_path),
  by = "label", suffix = c("_ues", "_topk")) %>%
  mutate(update = join_path_ues != join_path_topk)
join_path_updates %>% filter(update)
sum(join_path_updates$update)

# calculate the runtime difference between UES/Top-k variant for updated queries
approx_updated <- semi_join(df_approx %>% filter(subquery_mode == "smart"),
                            join_path_updates %>% filter(update),
                            by = c("label", "setting"))
update_runtimes <- inner_join(
  approx_updated %>% select(label, setting, execution_time),
  df_approx %>% filter(setting == "UES", subquery_mode == "smart") %>% select(label, execution_time),
  by = "label", suffix = c("_ues", "_topk")) %>%
  mutate(runtime_diff = execution_time_ues - execution_time_topk)
hist(update_runtimes$runtime_diff)
max(update_runtimes$runtime_diff)
min(update_runtimes$runtime_diff)

# determine the correlation between UES / Top-50 bounds and Top-50 bounds / True cardinalities
df_top50 <- df_approx %>% filter(setting == "Top-50", subquery_mode == "smart")
df_ues <- df_approx %>% filter(setting == "UES", subquery_mode == "smart")
bound_comparison <- inner_join(df_top50 %>% select(label, upper_bound, true_card), df_ues %>% select(label, upper_bound),
                               by = "label", suffix = c("_topk", "_ues"))
bound_comparison$tightening_factor <- bound_comparison$upper_bound_ues / bound_comparison$upper_bound_topk
max(bound_comparison$tightening_factor)
cor(bound_comparison$upper_bound_topk, bound_comparison$upper_bound_ues)
cor(bound_comparison$upper_bound_topk, bound_comparison$true_card)

# - - - - - - - - - - - - - - - -
# 14: Upper bounds for best approximative setting ----
# - - - - - - - - - - - - - - - -
df_top50 <- df_approx %>%
  filter(setting == "Top-50", subquery_mode == "linear", true_card > 0) %>%
  select(label, setting, upper_bound) %>%
  arrange(upper_bound)
df_ues <- df_approx %>%
  filter(setting == "UES", subquery_mode == "linear", true_card > 0) %>%
  select(label, setting, upper_bound)
df_true <- df_approx %>%
  filter(setting == "UES", subquery_mode == "linear", true_card > 0) %>%
  select(label, setting, true_card) %>%
  rename(upper_bound = true_card) %>%
  mutate(setting = "True cardinality")

df_plt <- bind_rows(df_top50, df_ues, df_true)
df_plt$label <- factor(df_plt$label, levels = df_top50$label, ordered = TRUE)

ggplot(df_plt, aes(x = label, y = upper_bound, color = setting, group = setting)) +
  geom_line() +
  geom_point() +
  labs(x = "Query (ordered by Top-50 upper bound)",  y = "Upper bound", color = "Workload") +
  scale_y_log10() +
  scale_color_viridis(option = "cividis", discrete = TRUE, end = 0.95) +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank())
ggsave("evaluation/plot-job-line-bounds-top50-approx.pdf")


# - - - - - - - - - - - - - - - -
# 15: Optimization time of the approximate algorithm ----
# - - - - - - - - - - - - - - - -
df_plt <- df_approx %>%
  filter(subquery_mode == "linear") %>%
  group_by(setting) %>%
  summarise(total_optimization_time = sum(optimization_time))
ggplot(df_plt, aes(x = setting, y = total_optimization_time, group = 1)) +
  geom_line(linetype = "dotted") +
  geom_point() +
  labs(x = "Workload", y = "Total optimization time [seconds]") +
  theme_bw()


# - - - - - - - - - - - - - - - -
# 15: Optimization time of the approximate algorithm ----
# - - - - - - - - - - - - - - - -

select_best_query_repetition <- function(result_path) {
  df <- read_csv(result_path)
  representatives <- df %>% group_by(label) %>% arrange(query_rt_total) %>% slice_head() %>% ungroup()
  return(representatives)
}

df_ssb <- bind_rows(
  select_best_query_repetition("workloads/ssb-results-implicit.csv") %>%
    select(label, query_rt_total) %>%
    rename(execution_time = query_rt_total) %>%
    mutate(setting = "implicit"),
  select_best_query_repetition("workloads/ssb-ues-results-base.csv") %>%
    select(label, ues_final_bound, optimization_time, query_rt_total) %>%
    rename(execution_time = query_rt_total, upper_bound = ues_final_bound) %>%
    mutate(setting = "UES"),
  select_best_query_repetition("workloads/topk-setups/ssb-ues-results-topk-15-approx.csv") %>%
    select(label, ues_final_bound, optimization_time, query_rt_total) %>%
    rename(execution_time = query_rt_total, upper_bound = ues_final_bound) %>%
    mutate(setting = "Top-15")
  )
ssb_true_cards <- read_csv("workloads/ssb-results-true-cards.csv") %>% select(label, query_result) %>% rename(true_card = query_result)

df_ssb$label <- factor(df_ssb$label, levels = unique(df_ssb$label), ordered = TRUE)
df_ssb$setting <- factor(df_ssb$setting, levels = c("implicit", "UES", "Top-15"), ordered = TRUE)
df_ssb <- inner_join(df_ssb, ssb_true_cards, by = "label")

ggplot(df_ssb, aes(x = label, y = execution_time, fill = setting)) +
  geom_col(position = "dodge") +
  labs(x = "Query", y = "Execution time [seconds]", fill = "Workload") +
  theme_bw() +
  theme(panel.grid.major.x = element_blank())
ggsave("evaluation/plot-ssb-runtimes.pdf")

df_plt <- bind_rows(df_ssb %>%
                      filter(setting != "implicit", true_card > 0) %>%
                      select(label, setting, upper_bound),
                    ssb_true_cards %>%
                      filter(true_card > 0) %>%
                      rename(upper_bound = true_card) %>%
                      mutate(setting = "True cardinality"))
ggplot(df_plt, aes(x = label, y = upper_bound, group = setting, color = setting)) +
  geom_point() +
  geom_line(linetype = "dotted") +
  labs(x = "Query", y = "Upper bound", color = "Workload") +
  scale_y_log10() +
  scale_color_viridis(option = "cividis", discrete = TRUE, end = 0.95) +
  theme_bw()
ggsave("evaluation/plot-ssb-bounds.pdf")

# - - - - - - - - - - - - - - - -
# XX: Test area ----
# - - - - - - - - - - - - - - - -
df_ues <- df_approx %>% filter(setting == "UES", subquery_mode == "smart")
df_top50 <- df_approx %>% filter(setting == "Top-50", subquery_mode == "smart")
df_rt <- inner_join(df_ues %>% select(label, execution_time),
                    df_top50 %>% select(label, execution_time),
                    by = "label",
                    suffix = c("_ues", "_top50"))
df_rt$runtime_diff <- df_rt$execution_time_ues - df_rt$execution_time_top50
summary(df_rt$runtime_diff)
hist(df_rt$runtime_diff)
