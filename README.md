# Fibration symmetries on Python

This is a python implementation of https://github.com/makselab/FastFibration fast fibration partitioning (FFP) algorithm

Initally, the algorithm was presented in [Morone et. al.](https://www.pnas.org/content/117/15/8306) in 2021. In 2025, FFP was used extensively in [Symmetries of Living Systems: Symmetry Fibrations and Synchronization in Biological Networks](https://arxiv.org/pdf/2502.18713)

## Usage 

To find the fibration partitioning of a given directed network it is necessary only the information of the network structure (nodes and edges)
and the types of each edge (in case the edges are multidimensional). For this, the network must be parsed as an **CSV file** following the structure
of an edgelist containing two essential informations (source and target) and one optional information (the type of the edge for multiplex scenarios). 
For instance, let us consider the graph below where the edges can assume two possible values: 'positive' or 'negative'.

<img src="small_example.png" width="500" />

The edgefile for this graph, called `net.csv` should follow the format below:

> Source,Target,Type<br/>
> 1,2,positive<br/>
> 2,1,positive<br/>
> 3,1,positive<br/>
> 3,4,positive<br/>
> 4,2,negative<br/>
> 4,3,positive<br/>
> 4,5,positive<br/>
> 6,3,negative<br/>
> 7,4,negative<br/>
> 8,6,positive<br/>
> 8,7,positive<br/>

In this file, the third column refers to the possible values of each edge. There is no restriction on the specific
format of its values as long as each different string represent a different edge type. For the first (source) and 
second (target) columns the node labels must be inside the interval \[1,N\] where N is the total number of nodes in
the graph.

Thus, to extract the fibers of the network provided by this edgefile, we run the following: 

```
graph,df = graph_from_csv("Graphs/net.csv", is_directed=True)
fiber_partition = fast_fibration(graph)
```

## Getting started
Run test.py to get started, then run fiber.py

The Graphs subdirectory contains:
- graphs with strongly connected components (SCC): 3 components (3SCC), 6 components (6SCC)
- graphs depicted in **Symmetries of Living Systems: Symmetry Fibrations and Synchronization in Biological Networks**: figure 1.3 page 10, figure 4.2 page 49, figure 17.5 page 358
- graph of E.Coli metabolism (test_Ecoli.txt)

## License

                  GNU LESSER GENERAL PUBLIC LICENSE
                       Version 2.1, February 1999

 Copyright (C) 1991, 1999 Free Software Foundation, Inc.
 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
