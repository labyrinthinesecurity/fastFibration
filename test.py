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
        graph_test_1,df = graph_from_csv("Graphs/test_1.csv", is_directed=True)
        graph_test3scc,df = graph_from_csv("Graphs/test_3SCC.csv", is_directed=True)
        graph_test3scc2,df = graph_from_csv("Graphs/test_3SCC_2.csv", is_directed=True)
        graph_test6scc,df = graph_from_csv("Graphs/test_6SCC.csv", is_directed=True)

        # Run the initialize function on test graphs
        part_test1, pq_test1 = initialize(graph_test_1)
        part_test3scc, pq_test3scc = initialize(graph_test3scc)
        part_test3scc2, pq_test3scc2 = initialize(graph_test3scc2)
        part_test6scc, pq_test6scc = initialize(graph_test6scc)

        # Assert partition and pivot queue are not empty
        self.assertTrue(part_test1)
        self.assertTrue(pq_test1)
        self.assertTrue(part_test3scc)
        self.assertTrue(pq_test3scc)
        self.assertTrue(part_test3scc2)
        self.assertTrue(pq_test3scc2)
        self.assertTrue(part_test6scc)
        self.assertTrue(pq_test6scc)

if __name__ == "__main__":
    unittest.main()

