#!/usr/bin/python3
import sys
from fast_fibration import *
graph,df = graph_from_csv("Graphs/test_Ecoli.csv", is_directed=True)
fiber_partition = fast_fibration(graph)
number_nontrivial_fibers, total_fibers, fiber_map = extract_fiber_groups(fiber_partition)
dump_fibration_base_to_csv(graph, fiber_map, fiber_map, "test_Ecoli.csv")
print("Number of fibers containing more than one node (nontrivial)",number_nontrivial_fibers-1)
#print(fiber_map)
#generate_graph_from_csv("base","net.csv")
