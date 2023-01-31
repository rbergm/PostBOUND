
from postbound.qal import qal
from postbound.optimizer.bounds import joins as join_bounds, scans as scan_bounds, subqueries, stats

def optimize_join_order(query: qal.ImplicitSqlQuery,
                        base_table_estimation: scan_bounds.BaseTableCardinalityEstimator,
                        join_estimation: join_bounds.JoinBoundCardinalityEstimator,
                        subquery_policy: subqueries.SubqueryGenerationPolicy,
                        stats_container: stats.UpperBoundsContainer):
    pass
