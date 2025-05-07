#!/usr/bin/python3
import sys
from fast_fibration import *

graph,df = graph_from_csv("Graphs/14.3p269.csv", is_directed=True)
#graph = graph_from_csv("Graphs/test_3SCC.csv", is_directed=True)
fiber_partition = fast_fibration(graph)
number_nontrivial_fibers, total_fibers, fiber_map = extract_fiber_groups(fiber_partition)
dump_fibration_base_to_csv(graph, fiber_map, fiber_map, "14.3p269.csv")
print("number of fibers containing more than one node (nontrivial)",number_nontrivial_fibers)
print("total number of fibers (including single node fibers)",total_fibers)
print(fiber_map)
generate_graph_from_csv("base","14.3p269.csv")
