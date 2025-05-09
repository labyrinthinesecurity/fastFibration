#!/usr/bin/python3
import unittest
from fast_fibration import Vertex, graph_from_csv, initialize

class TestFastFibration(unittest.TestCase):
    def test_vertex(self):
        # Testing Vertex object
        v = Vertex(0)
        self.assertEqual(v.index, 0)
        self.assertEqual(len(v.edges_source), 0)
        self.assertEqual(len(v.edges_target), 0)

    def test_initialize_algorithm(self):
        # Load test graphs from CSV files
        graph_test_1,df = graph_from_csv("Graphs/net.csv", is_directed=True)
        graph_test3scc,df = graph_from_csv("Graphs/test_3SCC.csv", is_directed=True)

        # Run the initialize function on test graphs
        part_test1, pq_test1 = initialize(graph_test_1)
        part_test3scc, pq_test3scc = initialize(graph_test3scc)

        # Assert partition and pivot queue are not empty
        self.assertTrue(part_test1)
        self.assertTrue(pq_test1)
        self.assertTrue(part_test3scc)
        self.assertTrue(pq_test3scc)

if __name__ == "__main__":
    unittest.main()

