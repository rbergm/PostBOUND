## plots.presentation.R
# Generates the plots used for our BTW'23 paper
#

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(forcats)
library(tidyr)
library(stringr)

library(gtools)

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

# 01: Runtime of the cautious algorithm ----
df_plt <- df_cautious %>% group_by(setting) %>% summarise(total_runtime = sum(execution_time))
ggplot(df_plt, aes(x = setting, y = total_runtime)) + 
  geom_col() +
  labs(x = "Workload", y = "Total runtime [seconds]") +
  theme_bw()
ggsave("evaluation/plot-job-runtimes-topk-cautious.pdf")

# 02: Median upper bounds of the cautious algorithm ----
df_plt <- df_cautious %>% group_by(setting) %>% summarise(median_bound = median(upper_bound))
ggplot(df_plt, aes(x = setting, y = median_bound, group = 1)) +
  geom_line(linetype = "dotted") +
  geom_point() +
  labs(x = "Workload", y = "Median upper bound") +
  theme_bw()
ggsave("evaluation/plot-job-median-bounds-topk-cautious.pdf")

# 03: Optimization time of the cautious algorithm ----
df_plt <- df_cautious %>% group_by(setting) %>% summarise(total_optimization_time = sum(optimization_time))
ggplot(df_plt, aes(x = setting, y = total_optimization_time, group = 1)) +
  geom_line(linetype = "dotted") +
  geom_point() +
  labs(x = "Workload", y = "Total optimization time [seconds]") +
  theme_bw()

# 11: runtime of the approximative algorithm ----
df_plt <- df_approx %>% group_by(setting, subquery_mode) %>% summarise(total_runtime = sum(execution_time))
ggplot(df_plt, aes(x = setting, y = total_runtime, fill = subquery_mode)) + 
  geom_col(position = "dodge") +
  labs(x = "Workload", y = "Total runtime [seconds]",
       fill = "Subquery generation") +
  scale_fill_viridis(option = "cividis", discrete = TRUE) +
  theme_bw()

# 12: Median upper bounds of the approximative algorithm ----
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

# 13: Upper bounds for best approximative setting ----
df_top50 <- df_approx %>% 
  filter(setting == "Top-50", subquery_mode == "linear", true_card > 0) %>%
  select(label, setting, upper_bound)
df_ues <- df_approx %>%
  filter(setting == "UES", subquery_mode == "linear", true_card > 0) %>%
  select(label, setting, upper_bound)
df_true <- df_approx %>%
  filter(setting == "UES", subquery_mode == "linear", true_card > 0) %>%
  select(label, setting, true_card) %>%
  rename(upper_bound = true_card) %>%
  mutate(setting = "True cardinality") %>%
  arrange(upper_bound)

df_plt <- bind_rows(df_top50, df_ues, df_true)
df_plt$label <- factor(df_plt$label, levels = df_true$label, ordered = TRUE)

ggplot(df_plt, aes(x = label, y = upper_bound, color = setting, group = setting)) +
  geom_line() +
  geom_point() +
  scale_y_log10() +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())


# XX: Test area ----
df_ues <- df_approx %>% filter(setting == "UES", subquery_mode == "smart")
df_top50 <- df_approx %>% filter(setting == "Top-50", subquery_mode == "smart")
df_rt <- inner_join(df_ues %>% select(label, execution_time),
                    df_top50 %>% select(label, execution_time),
                    by = "label",
                    suffix = c("_ues", "_top50"))
df_rt$runtime_diff <- df_rt$execution_time_ues - df_rt$execution_time_top50
summary(df_rt$runtime_diff)
hist(df_rt$runtime_diff)
