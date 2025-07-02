import postbound as pb

workload = pb.workloads.job()
query = workload["1a"]

print("===== Raw query =====\n")
print(pb.qal.format_quick(query))
print()

print("===== Tables =====\n")
for table in query.tables():
    print("-", table.full_name)
print()

print("===== Filter predicates =====\n")
for predicate in query.predicates().filters():
    print("-", predicate)

    # Accessing the raw predicate's contents can be rather cumbersome, because we need to represent arbitrarily complex SQL
    # expressions. For query optimization, we often focus on simpler cases, e.g. filters that follow the form
    # <column> <op> <value>.
    # We can work with them more easily by using the SimpleFilter

    if not pb.qal.SimpleFilter.can_wrap(predicate):
        # First, we need to make sure that our predicate is actually of such a simple form!
        continue

    simplified = pb.qal.SimpleFilter.wrap(predicate)
    print(
        "-", simplified.column, simplified.operation, simplified.value, "(simplified)"
    )

print()

print("===== Join predicates =====\n")
for predicate in query.predicates().joins():
    print("-", predicate)
print()
