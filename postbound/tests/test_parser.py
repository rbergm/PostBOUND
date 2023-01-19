
import unittest

import mo_sql_parsing as mosp
import mo_parsing.exceptions

import regression_suite


class StackWorkloadTests(unittest.TestCase):
    def test_parse_format(self):
        parse_exceptions = {}
        err_msgs = set()
        success = 0
        stack_workload = regression_suite.load_stack_workload()
        for label, query in stack_workload.items():
            try:
                parsed = mosp.parse(query)
                reformatted = mosp.format(parsed)  # noqa: F841
                success += 1
            except mo_parsing.exceptions.ParseException as e:
                #self.fail(f"ParseException at label {label}: {e}")
                parse_exceptions[label] = e
                err_msgs.add(str(e))

        print("\n".join(["Successfully optimized ::", str(success),
                         "ParseExceptions ::", str(len(parse_exceptions)),
                         "Exception conditions ::", "\n".join(err_msgs)]))
