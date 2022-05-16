# subquery-evaluation.R
# focused on the usage of subqueries..

library(functional)

library(readr)
library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(stringr)
library(jsonlite)
library(ggplot2)
library(viridis, quietly = TRUE)
library(scales, warn.conflicts = FALSE)

parse_json <- Vectorize(fromJSON, USE.NAMES = FALSE)

df <- read_csv("workloads/job-ues-eval-fks-nonlj.csv")
df$subquery_tables <- parse_json(df$subquery_tables)

# [+] Barplot :: Total runtime of UES / linearized workload ----
df_plt <- df %>%
  select(label, runtime_ues, runtime_flat) %>%
  rename(ues = runtime_ues, linearized = runtime_flat) %>%
  pivot_longer(cols = c(ues, linearized),
               names_to = "workload", values_to = "runtime") %>%
  group_by(workload) %>%
  summarise(runtime = sum(runtime))
ggplot(df_plt, aes(x = workload, y = runtime, fill = workload)) +
  geom_col() +
  geom_text(aes(label = runtime), vjust = -.3) +
  labs(title = "Total runtime of the UES / linearized workloads",
       x = "Workload",  y = "Total runtime [seconds]", fill = "Workload") +
  scale_fill_viridis(discrete = TRUE, option = "cividis") +
  theme_bw()


# Boxplot :: runtime of UES / linearized workload ----
df_plt <- df %>%
  select(label, runtime_ues, runtime_flat) %>%
  rename(ues = runtime_ues, linearized = runtime_flat) %>%
  pivot_longer(cols = c(ues, linearized),
               names_to = "workload", values_to = "runtime")
ggplot(df_plt, aes(x = workload, y = runtime, fill = workload)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, option = "cividis") +
  labs(x = "Workload", y = "Runtime [seconds]", fill = "Workload") +
  theme_bw()

# [+] Lineplot :: absolute runtime differences of UES / linearized workload ----
df_plt <- df %>%
  mutate(runtime_diff = runtime_flat - runtime_ues,
         ues_faster = ifelse(runtime_diff > 0, "UES", "linearized")) %>%
  arrange(desc(runtime_diff))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = runtime_diff, fill = ues_faster, color = ues_faster)) +
  geom_line() +
  geom_area() +
  labs(title = "Absolute runtime difference between UES / linearized queries",
       x = "Query (ordered by runtime difference)",  y = "Runtime difference [seconds]",
       fill = "Faster query", color = "Faster query") +
  scale_fill_viridis(discrete = TRUE, option = "cividis") +
  scale_color_viridis(discrete = TRUE, option = "inferno") +
  theme_bw() +
  theme(axis.text.x = element_blank())

# Lineplot :: relative runtime differences of UES / linearized workload ----
df_plt <- df %>%
  mutate(runtime_diff = runtime_flat / runtime_ues,
         ues_faster = ifelse(runtime_diff > 1, "UES", "linearized")) %>%
  arrange(desc(runtime_diff))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = runtime_diff, fill = ues_faster, color = ues_faster)) +
  geom_line() +
  geom_ribbon(aes(ymin = 1, ymax = df_plt$runtime_diff)) +
  labs(title = "Relative runtime difference between UES / linearized queries",
       x = "Query (ordered by runtime difference)",  y = "Runtime difference [factor]",
       fill = "Faster query", color = "Faster query") +
  scale_fill_viridis(discrete = TRUE, option = "cividis") +
  scale_color_viridis(discrete = TRUE, option = "inferno") +
  theme_bw() +
  theme(axis.text.x = element_blank())

# [+] Boxplot :: filter strength distribution ----
ggplot(df, aes(x = "all queries", y = filter_strength)) +
  geom_boxplot() +
  scale_y_log10() +
  stat_summary(geom = "text", fun = quantile,
               aes(label = sprintf("10^%1.1f", ..y..)),
               position = position_nudge(x = 0.41)) +
  labs(title = "Filter strength distribution", y = "Filter strength") +
  theme_bw() +
  theme(axis.title.x = element_blank(), axis.text.x =  element_blank())

# Boxplot :: absolute filter strength distribution ----
df_plt <- df %>% mutate(filter_strength = foreign_key_rows - rows_after_join)
ggplot(df_plt, aes(x = "all queries", y = filter_strength)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "Distribution of absolute filter strength", y = "Filter strength") +
  theme_bw() +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  
# [+] Scatterplot :: Correlation between filter strength / rel. speedup ----
df_plt <- df %>%
  filter(!is.na(filter_strength)) %>%
  mutate(branch_runtime_diff = subquery_partner_runtime - subquery_runtime,
         pruned_status = ifelse(ues_pruned & flat_pruned, "both",
                                ifelse(ues_pruned, "UES",
                                       ifelse(flat_pruned, "linearized", "none"))))
ggplot(df_plt, aes(x = filter_strength, y = ues_speedup,
                   color = branch_runtime_diff, shape = pruned_status)) +
  geom_point() +
  scale_x_log10(breaks = breaks_log(), labels = label_scientific()) +
  labs(title = "Correlation between subquery filter strength and observed relative speedup",
       subtitle = paste("Filter strength = ratio of incoming/outgoing rows in subquery",
                        "Speedup = ratio of linearized runtime to UES runtime", sep = "\n"),
       x = "Filter strength", y = "UES speedup",
       color = "Speedup of subquery compared\nto partner branch [seconds]",
       shape = "Nodes pruned") +
  scale_color_viridis() +
  theme_bw()
ggsave("evaluation/corr-filter-speedup-rel.pdf")


# Scatterplot :: Correlation between filter strength / abs. speedup ----
df_plt <- df %>%
  filter(!is.na(filter_strength)) %>%
  mutate(ues_speedup = runtime_flat - runtime_ues,
         branch_runtime_diff = subquery_partner_runtime - subquery_runtime,
         pruned_status = ifelse(ues_pruned & flat_pruned, "both",
                                ifelse(ues_pruned, "UES",
                                       ifelse(flat_pruned, "linearized", "none"))))
ggplot(df_plt, aes(x = filter_strength, y = ues_speedup,
                   color = branch_runtime_diff, shape = pruned_status)) +
  geom_point() +
  scale_x_log10(breaks = breaks_log(), labels = label_scientific()) +
  labs(title = "Correlation between subquery filter strength and observed absolute speedup",
       subtitle = paste("Filter strength = ratio of incoming/outgoing rows in subquery",
                        "Speedup = Difference between linearized / UES runtime", sep = "\n"),
       x = "Filter strength", y = "UES speedup",
       color = "Speedup of subquery compared\nto partner branch [seconds]",
       shape = "Nodes pruned (UES plan)") +
  scale_color_viridis() +
  theme_bw()
ggsave("evaluation/corr-filter-speedup-abs.pdf")

# [+] Scatterplot :: Correlation between abs. filter strength / abs. speedup ----
df_plt <- df %>%
  mutate(filter_strength = foreign_key_rows - rows_after_join) %>%
  filter(!is.na(filter_strength), filter_strength != 0) %>%
  mutate(ues_speedup = runtime_flat - runtime_ues,
         branch_runtime_diff = subquery_partner_runtime - subquery_runtime,
         pruned_status = ifelse(ues_pruned & flat_pruned, "both",
                                ifelse(ues_pruned, "UES",
                                       ifelse(flat_pruned, "linearized", "none"))))
ggplot(df_plt, aes(x = filter_strength, y = ues_speedup,
                   color = branch_runtime_diff, shape = pruned_status)) +
  geom_point() +
  scale_x_log10(breaks = breaks_log(), labels = label_scientific()) +
  labs(title = "Correlation between subquery filter strength and observed absolute speedup",
       subtitle = paste("Filter strength = Difference between number of incoming vs. outgoing rows in subquery",
                        "Speedup = Difference between linearized vs. UES runtime", sep = "\n"),
       x = "Filter strength", y = "UES speedup",
       color = "Speedup of subquery compared\nto partner branch [seconds]",
       shape = "Applied pruning actions\nper query variant") +
  scale_color_viridis() +
  theme_bw()
ggsave("evaluation/corr-abs-filter-abs-speedup.pdf")

# [+] Lineplot :: absolute runtime difference between subquery / partner branch ----
df_plt <- df %>%
  filter(!is.na(subquery_runtime), !is.na(subquery_partner_runtime)) %>%
  mutate(branch_runtime_diff = subquery_partner_runtime - subquery_runtime,
         faster_subquery = ifelse(branch_runtime_diff > 0, "Subquery", "Join partner")) %>%
  arrange(desc(branch_runtime_diff))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = branch_runtime_diff, fill = faster_subquery, color = faster_subquery)) +
  geom_line(linetype = "dotted", size = 0.7) +
  geom_area() +
  labs(title = "Runtime difference between subquery and partner branch",
       subtitle = "Partner branch = query subtree that gets joined with the subquery result",
       x = "Query (ordered by runtime difference)", y = "Runtime difference [seconds]",
       fill = "Faster subtree", color = "Faster subtree") +
  scale_fill_viridis(discrete = TRUE, option = "cividis") +
  scale_color_viridis(discrete = TRUE, option = "inferno") +
  theme_bw() +
  theme(axis.text.x = element_blank())

# [+] Scatterplot :: correlation between subquery speedup / total speedup ----
df_plt <- df %>%
  filter(!is.na(subquery_runtime), !is.na(subquery_partner_runtime), !is.na(ues_speedup)) %>%
  mutate(branch_runtime_diff = subquery_partner_runtime - subquery_runtime,
         total_speedup = runtime_flat - runtime_ues) %>%
  arrange(desc(branch_runtime_diff))
ggplot(df_plt, aes(x = branch_runtime_diff, y = total_speedup, color = log(filter_strength))) +
  geom_point() +
  labs(title = "Correlation between subquery speedup and total speedup",
       subtitle = paste("Subquery speedup = difference between join partner runtime and subquery runtime",
                        "Total speedup = difference of linearized runtime and UES runtime", sep = "\n"),
       x = "Subquery speedup [seconds]", y = "Total speedup [seconds]",
       color = "Filter strength [log]") +
  scale_color_viridis(option = "cividis") +
  theme_bw()

# [+] Line/Scatterplot :: "correlation" between pruning + speedup ----
df_plt <- df %>%
  mutate(ues_abs_speedup = runtime_flat - runtime_ues,
         pruned_status = ifelse(ues_pruned & flat_pruned, "both",
                                ifelse(ues_pruned, "UES",
                                       ifelse(flat_pruned, "linearized", "none")))) %>%
  arrange(desc(ues_abs_speedup))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = ues_abs_speedup,
                   shape = pruned_status, color = pruned_status)) +
  geom_point() +
  labs(title="'Correlation' between pruning and achieved speedup",
       x = "Query (ordered by speedup)", y = "Absolute UES speedup [seconds]",
       color = "Pruned nodes", shape = "Pruned nodes") +
  scale_color_viridis(discrete = TRUE, option = "cividis") +
  theme_bw() +
  theme(axis.text.x = element_blank())
ggsave("evaluation/corr-pruning-speedup-abs.pdf")

# [+] Scatterplot :: Correlation between referenced tables and total speedup ----
df_plt <- df %>%
  filter(!is.na(ues_speedup), !is.na(filter_strength)) %>%
  unnest(subquery_tables)
ggplot(df_plt, aes(x = subquery_tables, y = ues_speedup, color = log(filter_strength))) +
  geom_point() +
  labs(title = "UES speedup per referenced table",
       x = "Table referenced in subquery",  y = "UES speedup", color = "Filter strength [log]") +
  theme_bw() +
  scale_color_viridis(option = "cividis") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(df_plt, aes(x = subquery_tables, y = ues_speedup, color = label)) +
  geom_point() +
  labs(x = "Table referenced in subquery",  y = "UES speedup", color = "Query label") +
  theme_bw() +
  scale_color_viridis(discrete = TRUE, option = "cividis") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# [+] KMeans ----

min_max_scaler <- function(data) {
  dmin <- min(ifelse(is.finite(data), data, Inf), na.rm = TRUE)
  dmax <- max(ifelse(is.finite(data), data, -Inf), na.rm = TRUE)
  scaled <- (data - dmin) / (dmax - dmin)
  return(scaled)
}

df_clust <- df %>%
  mutate(ues_speedup = runtime_flat - runtime_ues,
         branch_runtime_diff = subquery_partner_runtime - subquery_runtime,
         filter_strength = ifelse(is.finite(filter_strength), filter_strength, 0)) %>%
  mutate(ues_speedup = min_max_scaler(ues_speedup),
         branch_runtime_diff = min_max_scaler(branch_runtime_diff),
         filter_strength = min_max_scaler(filter_strength)) %>%
  select(ues_speedup, branch_runtime_diff, ues_pruned, flat_pruned, subquery_pruned, filter_strength)
df_clust$cluster <- kmeans(df_clust, 2)$cluster
df_clust$label <- df$label

df_plt <- df_clust %>%
  mutate(ues_speedup = df$ues_speedup, cluster = as.factor(cluster)) %>%
  arrange(desc(ues_speedup))
ggplot(df_plt, aes(x = 1:nrow(df_plt), y = ues_speedup, color = cluster)) +
  geom_point() +
  scale_color_viridis(discrete = TRUE, option = "cividis") +
  labs(title = "Clusters associated to JOB queries",
       x = "Query (ordered by speedup)", y = "UES speedup [seconds]",
       color = "Cluster") +
  theme_bw()

# [+] Scatterplot :: Correlation between Subquery runtime and speedup ----
df_plt <- df %>% mutate(ues_speedup = runtime_flat - runtime_ues)
ggplot(df_plt, aes(x = subquery_runtime, y = ues_speedup,
                   color = log(filter_strength))) +
  geom_point() +
  geom_smooth(method = lm, color = "#7a7a77", se = FALSE, size = 0.3) +
  scale_color_viridis(option = "cividis") +
  labs(title = "Correlation between subquery runtime and speedup",
       x = "Subquery runtime [seconds]", y = "UES speedup [seconds]",
       color = "Filter strength [log]") +
  theme_bw()
