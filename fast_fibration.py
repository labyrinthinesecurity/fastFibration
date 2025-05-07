#!/usr/bin/python3

import pandas as pd
import csv
import numpy as np
from collections import defaultdict, deque
import random,sys,os,json
import math
from pyvis.network import Network
VERBOSE=False

cache={}
nodenamescache={}
spnscache={}

# the following statement is only used in Azure fibrations, 
# it can be safely ignored in all other scenarios
if os.path.exists("spns_cache.json"):
  with open("spns_cache.json", "r") as file:
    spnscache=json.load(file)


def generate_graph_from_csv(what,filename):
  COLORS = ["#435055", "#7694a1", "#a5c3ab", "#ffa500", "#ffde21"]
  if what=='base':
    df = pd.read_csv("Bases/"+filename)
  elif what=='graph':
    df = pd.read_csv("Graphs/"+filename)
  else:
    print("ERR: what must be base or graph")
    sys.exit()
  graph = Network(notebook=True, directed=True, cdn_resources='in_line',height='1000px')
#  graph.force_atlas_2based(gravity=-50, central_gravity=0.000, spring_length=150, spring_strength=0.007)
  graph.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=150, spring_strength=0.05)
  edge_counts = {}
  self_loops = {}
  for _, row in df.iterrows():
    source, target, edge_type = row["SourceName"], row["TargetName"], row["EdgeType"]
    if source in spnscache:
      graph.add_node(source, label=source,physics=True, color=COLORS[4])
    else: 
      graph.add_node(source, label=source,physics=True, color=COLORS[1])
    if target in spnscache:
      graph.add_node(target, label=target,physics=True, color=COLORS[4])
    else: 
      graph.add_node(target, label=target,physics=True, color=COLORS[1])
    edge_color = "green" if edge_type == "plus" else "red"
    # Handle self-loops explicitly
    if source == target:
        if source in self_loops:
            self_loops[source] += 1
        else:
            self_loops[source] = 1
        graph.add_edge(source, target, color=edge_color, width=0.5, dashes=[2,5], physics=True, smooth={"type": "curvedCW"}, selfReferenceSize=20 + 10 * self_loops[source])
    else:
        # Handle duplicate edges
        edge_key = (source, target)
        if edge_key in edge_counts:
            edge_counts[edge_key] += 1
        else:
            edge_counts[edge_key] = 1
        smooth_type = ["curvedCW", "curvedCCW", "dynamic"][edge_counts[edge_key] % 3]
        graph.add_edge(source, target, color=edge_color, physics=True, smooth={"type": smooth_type})
  graph.show(f"{what}.html")

def dump_pseudofiber_base_to_csv(graph, node_ids, edges, filename):
    with open(filename, mode="w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(["SourceName", "TargetName", "EdgeType", "Source", "Target"])
      for edge in edges:
        cacheline=str(edge[0])+"->"+str(edge[1])
        if cacheline in cache:
#              print("HIT")
          pass
        else:
           cache[cacheline]=1
           writer.writerow([edge[0],edge[1], node_ids[edge[0]], node_ids[edge[1]]])

def dump_fibration_graph_to_csv(graph, filename):
    edgetype_prop = graph.string_eproperties["edgetype"]  # Edge types
#    print("BASE",base)
    with open("Graphs/"+filename, mode="w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(["SourceName", "TargetName", "EdgeType", "Source", "Target"])
      for n in range(0,graph.N):
        for edge in graph.vertices[n].edges_target:
          n1 = edge.source
          cacheline=str(n1+1)+"->"+str(n+1)+":"+edgetype_prop[edge.index]
          if cacheline in cache:
            pass
          else:
             cache[cacheline]=1
             writer.writerow([edge.sourcen, edge.targetn, edgetype_prop[edge.index], n1+1, n+1])

def dump_fibration_base_to_csv(graph, base, fiber_map, filename):
    edgetype_prop = graph.string_eproperties["edgetype"]  # Edge types
    namec={}
    for n in range(0,graph.N):
      for edge in graph.vertices[n].edges_target:
        if str(edge.source) not in namec:
          namec[str(edge.source)]=edge.sourcen
      for edge in graph.vertices[n].edges_source:
        if str(edge.target) not in namec:
          namec[str(edge.target)]=edge.targetn
    #for c in namec:
    #  print(c,namec[c])
#    print("BASE",base)
    with open("Bases/"+filename, mode="w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(["SourceName", "TargetName", "EdgeType", "Source", "Target"])
      for f in base:
        #print(f,base[f][0],base[f])
        n0 = base[f][0]  # node representative of fiber orbit
        for n in base[f]:
          for edge in graph.vertices[n].edges_target:
            src = edge.source
            srcn = edge.sourcen
            rep_src = None
            rep_tgt = None
            for g in fiber_map:
              if edge.source in fiber_map[g]:
                rep_src = fiber_map[g][0]
                if rep_tgt:
                  break
              if edge.target in fiber_map[g]:
                rep_tgt = fiber_map[g][0]
                if rep_src:
                  break
            if rep_src is None or (rep_tgt is None):
              print("warning dumping base to csv... ",rep_src,rep_tgt)
              sys.exit()
            else:
              cacheline=namec[str(rep_src)]+"->"+namec[str(rep_tgt)]+":"+edgetype_prop[edge.index]
            if cacheline in cache:
              pass
            else:
               cache[cacheline]=1
               writer.writerow([namec[str(rep_src)],namec[str(rep_tgt)], edgetype_prop[edge.index], rep_src+1, rep_tgt+1])
# graph_def.jl

class Edge:
    """
    Defines the structure for an edge of a graph.
    It is defined by its 'index', 'source', and 'target'.

    'source' and 'target' refer to the node at the tail of the edge and 
    the node at the head of the edge, respectively.
    """
    def __init__(self, source: int, target: int, srcn: str, tgtn: str):
        self.index = -1
        self.source = source
        self.target = target
        self.sourcen = srcn
        self.targetn = tgtn


class Vertex:
    """
    Defines the structure for a vertex/node of a graph.
    It is defined by its 'index', 'edges_source', and 'edges_target'.
    """
    def __init__(self, index: int):
        self.index = index
        self.edges_source = []  # List of Edge objects
        self.edges_target = []  # List of Edge objects

# fiber_def.jl

class Fiber:
    """
    Struct to define the properties of a fiber object.

    Properties:
        index (int): Index during the algorithms.
        number_nodes (int): Size of the fiber.
        number_regulators (int): Number of fiber regulators.
        regulators (list[int]): Indexes of the regulators of the fiber.
        nodes (list[int]): Indexes of the nodes belonging to the fiber.
        input_type (dict[int, int]): Key is a type of input, value is
                                     the number of this type of input.
    """
    def __init__(self):
        self.index = 0
        self.number_nodes = 0
        self.number_regulators = 0
        self.regulators = []
        self.nodes = []
        self.input_type = {}

    def insert_nodes(self, nodelist):
        if isinstance(nodelist, int):
            self.nodes.append(nodelist)
        elif isinstance(nodelist, list):
            self.nodes.extend(nodelist)
        self.number_nodes = len(self.nodes)

    def delete_nodes(self, nodelist):
        self.nodes = [node for node in self.nodes if node not in nodelist]
        self.number_nodes = len(self.nodes)

    def insert_regulator(self, reg: int):
        self.regulators.append(reg)
        self.number_regulators += 1

    def get_nodes(self):
        return self.nodes

    def copy(self):
        new_fiber = Fiber()
        new_fiber.nodes = self.nodes.copy()
        new_fiber.index = self.index
        return new_fiber

    def get_number_nodes(self):
        return len(self.nodes)


def successor_nodes(graph, fiber):
    """
    Returns all nodes in 'graph' that are pointed to by 'fiber'.
    """
    successor = []
    for node in fiber.nodes:
        out_neigh = get_out_neighbors(node, graph)
        successor.extend(out_neigh)
    return list(set(successor))


def input_stability(fiber, pivot, graph, num_edgetype):
    """
    Given two Fiber objects 'fiber' and 'pivot', checks if 'fiber' is
    input-set stable with respect to 'pivot'.

    If input-set stable, returns True. Otherwise, returns False.
    """
    fiber_nodes = fiber.get_nodes()
    pivot_nodes = pivot.get_nodes()
    edges_received = {}

    edgelist = graph.edges
    edgetype = graph.int_eproperties["edgetype"]

    # Initialize input-set array for each node of 'fiber'
    for node in fiber_nodes:
        edges_received[node] = [0] * num_edgetype

    # Set input-set of each node of 'fiber' based on outgoing edges of 'pivot'
    for w in pivot_nodes:
        pivot_obj = graph.vertices[w]
        out_edges = pivot_obj.edges_source
        for out_edge in out_edges:
            edge_index = out_edge.index
            target_node = out_edge.target
            if target_node in edges_received:
                edges_received[target_node][edge_index] += 1

    # Check input-set stability
    for j in range(len(fiber_nodes) - 1):
        if edges_received[fiber_nodes[j]] != edges_received[fiber_nodes[j + 1]]:
            return False
    return True


def copy_fiber(fiber):
    """
    Returns a copy of a given Fiber object.
    """
    copy_fiber = Fiber()
    copy_fiber.index = fiber.index
    copy_fiber.nodes = fiber.nodes.copy()
    copy_fiber.input_type = fiber.input_type.copy()
    copy_fiber.number_nodes = len(copy_fiber.nodes)
    copy_fiber.number_regulators = fiber.number_regulators
    copy_fiber.regulators = fiber.regulators.copy()
    return copy_fiber


# strongcomp_def.jl

class StrongComponent:
    """
    Struct to define the properties of a strongly connected component (SCC) object.

    Properties:
        number_nodes (int): Size of the SCC.
        have_input (bool): Whether the SCC has input from another SCC.
        nodes (list[int]): Indexes of nodes belonging to the SCC.
        type (int): Classification of SCC (0 to 2) based on its inputs.
    """
    def __init__(self):
        self.number_nodes = 0
        self.have_input = False
        self.nodes = []
        self.type = -1

    def insert_nodes(self, node):
        """
        Add 'node' to 'strong' object. If 'node' is a list, insert all nodes.
        """
        if isinstance(node, int):
            self.nodes.append(node)
        elif isinstance(node, list):
            self.nodes.extend(node)
        self.number_nodes = len(self.nodes)

    def get_nodes(self):
        return self.nodes

    def get_input_bool(self):
        return self.have_input

    def check_input(self, graph):
        """
        Check if the given SCC receives input from another component in the 'graph'.
        Modifies 'have_input' to True if external input is detected.
        """
        for u in self.nodes:
            input_nodes = get_in_neighbors(u, graph)
            for w in input_nodes:
                if w not in self.nodes:
                    self.have_input = True
                    return

    def classify_strong(self, graph):
        """
        This function should be called after 'check_input'.
        Classifies the SCC based on external input and internal structure.
        """
        if self.have_input:
            self.type = 0
        else:
            # Check if it's an isolated self-loop node
            if len(self.nodes) == 1:
                in_neighbors = get_in_neighbors(self.nodes[0], graph)
                if len(in_neighbors) == 0:
                    self.type = 1
                else:
                    self.type = 2  # Isolated self-loop node
            else:
                self.type = 1  # SCC does not have external input


# graph_prop.jl

def set_vertices_properties(name, arr, graph):
    """
    Set a mapping property for the vertices.

    Depending on the type of elements in 'arr', the properties are set in different dictionaries.
    If a property with the given 'name' already exists, it will be replaced without warning.

    Args:
        name (str): Name of the property.
        arr (list): List of typed elements.
        graph (Graph): Graph object to assign the property to.
    """
    if len(graph.vertices) == len(arr):
        if all(isinstance(x, int) for x in arr):
            graph.int_vproperties[name] = arr
        elif all(isinstance(x, float) for x in arr):
            graph.float_vproperties[name] = arr
        elif all(isinstance(x, str) for x in arr):
            graph.string_vproperties[name] = arr
    else:
        print("size array does not match number of vertices")


def set_edges_properties(name, arr, graph):
    """
    Set a mapping property for the edges.

    Depending on the type of elements in 'arr', the properties are set in different dictionaries.
    If a property with the given 'name' already exists, it will be replaced without warning.

    Args:
        name (str): Name of the property.
        arr (list): List of typed elements.
        graph (Graph): Graph object to assign the property to.
    """
    if len(graph.edges) == len(arr):
        if all(isinstance(x, int) for x in arr):
            graph.int_eproperties[name] = arr
        elif all(isinstance(x, float) for x in arr):
            graph.float_eproperties[name] = arr
        elif all(isinstance(x, str) for x in arr):
            graph.string_eproperties[name] = arr
    else:
        print("size array does not match number of edges")


# basic_ops.jl

def get_edgelist(graph):
    """
    Returns an edge list as a list of (source, target) tuples.
    """
    return [(edge.source, edge.target) for edge in graph.edges]

def copy_graph(graph):
    """
    Create a copy of the graph.
    """
    new_graph = Graph(graph.is_directed)
    for node in graph.vertices:
        new_graph.add_node()
    for edge in graph.edges:
        new_graph.add_edge(edge.source, edge.target,edge.sourcen,edge.targetn)
    new_graph.int_vproperties = graph.int_vproperties.copy()
    new_graph.float_vproperties = graph.float_vproperties.copy()
    new_graph.string_vproperties = graph.string_vproperties.copy()
    new_graph.int_eproperties = graph.int_eproperties.copy()
    new_graph.float_eproperties = graph.float_eproperties.copy()
    new_graph.string_eproperties = graph.string_eproperties.copy()
    return new_graph

# io_ops.jl

def graph_from_csv(file_path, is_directed):
    """
    Create graph from CSV file.
    The file must contain columns: 'Source', 'Target', and optionally 'Type'.
    """
    df = pd.read_csv(file_path)
    graph = Graph(is_directed)
    N = max(df['Source'].max(), df['Target'].max())
    print("Max nodes N=",N)
    # Create the vertices
    for i in range(0, N):
        if VERBOSE:
          print("adding node",i)
        graph.add_node()

    # Add the edges
    if 'SourceName' in df and 'TargetName' in df:
      for src, tgt,srcn,tgtn in zip(df['Source'], df['Target'],df['SourceName'], df['TargetName']):
          #print("adding edge from src",src-1,"to tgt",tgt-1)
          if srcn not in nodenamescache:
            nodenamescache[str(src)]=srcn
          if tgtn not in nodenamescache:
            nodenamescache[str(tgt)]=tgtn
          graph.add_edge(src-1, tgt-1, srcn, tgtn)
    else:
      for src, tgt in zip(df['Source'], df['Target']):
          #print("adding edge from src",src-1,"to tgt",tgt-1)
          graph.add_edge(src-1, tgt-1, None, None)

    if 'Type' in df.columns:
        if df['Type'].dtype == np.int64:
            set_edges_properties("edgetype", df['Type'].tolist(), graph)
        elif df['Type'].dtype == object:
            unique_types = df['Type'].unique()
            label_map = {label: idx  for idx, label in enumerate(unique_types)}
            int_types = [label_map[val] for val in df['Type']]
            set_edges_properties("edgetype", int_types, graph)
            set_edges_properties("edgetype", df['Type'].tolist(), graph)
        else:
            raise ValueError("Format of 'Type' column is not accepted.")
    else:
        set_edges_properties("edgetype", [1] * graph.M, graph)

    return graph,df


# topology.jl

def get_in_neighbors(node, graph):
    node_obj = graph.vertices[node]
    incoming_neighbors = {edge.source for edge in node_obj.edges_target if edge.target == node}
    return list(incoming_neighbors)

def get_out_neighbors(node, graph):
    node_obj = graph.vertices[node]
    outgoing_neighbors = {edge.target for edge in node_obj.edges_source if edge.source == node}
    return list(outgoing_neighbors)

def bfs_search(source, graph):
    N = len(graph.vertices)
    color = [-1] * N
    dist = [-1] * N
    parent = [-1] * N

    color[source] = 0
    dist[source] = 0
    parent[source] = -1

    queue = deque([source])
    while queue:
        u = queue.popleft()
        for w in get_out_neighbors(u, graph):
            if color[w] == -1:
                color[w] = 0
                dist[w] = dist[u] + 1
                parent[w] = u
                queue.append(w)
        color[u] = 1
    return color, dist, parent

def dfs_search(graph):
    N = len(graph.vertices)
    color = [-1] * N
    dist = [-1] * N
    parent = [-1] * N
    finished = [-1] * N

    time = [0]
    for u in range(N):
        if color[u] == -1:
            dfs_visit(u, graph, time, color, dist, parent, finished)
    return color, parent, finished

def transpose_graph(graph):
    for edge in graph.edges:
        edge.source, edge.target = edge.target, edge.source
    for node in graph.vertices:
        node.edges_source, node.edges_target = node.edges_target, node.edges_source

def get_root(node, parent):
    while parent[node] != -1:
        node = parent[node]
    return node

def dfs_visit(u, graph, time, color, dist, parent, finished):
    time[0] += 1
    dist[u] = time[0]
    color[u] = 0

    for v in get_out_neighbors(u, graph):
        if color[v] == -1:
            parent[v] = u
            dfs_visit(v, graph, time, color, dist, parent, finished)

    color[u] = 1
    time[0] += 1
    finished[u] = time[0]

def extract_strong(graph, return_dict=False):
    N = len(graph.vertices)
    _, parent, finished = dfs_search(graph)

    graph_t = copy_graph(graph)
    transpose_graph(graph_t)

    time = [0]
    color_t = [-1] * N
    dist_t = [-1] * N
    parent_t = [-1] * N
    finished_t = [-1] * N

    node_ordering = np.argsort(finished)[::-1]
    for u in node_ordering:
        if color_t[u] == -1:
            dfs_visit(u, graph_t, time, color_t, dist_t, parent_t, finished_t)

    scc_trees = defaultdict(list)
    node_labels = [-1] * N
    for u in range(N):
        root = get_root(u, parent_t)
        scc_trees[root].append(u)
        node_labels[u] = root

    unique_labels = list(set(node_labels))
    return (node_labels, unique_labels, scc_trees) if return_dict else (node_labels, unique_labels)


# generation.jl

class Graph:
    def __init__(self, is_directed):
        self.N = 0  # Number of vertices
        self.M = 0  # Number of edges
        self.is_directed = is_directed
        self.vertices = []
        self.edges = []
        self.int_eproperties = {}
        # Vertex properties
        self.int_vproperties = {}  # Dict[str, list[int]]
        self.float_vproperties = {}  # Dict[str, list[float]]
        self.string_vproperties = {}  # Dict[str, list[str]]

        # Edge properties
        self.int_eproperties = {}  # Dict[str, list[int]]
        self.float_eproperties = {}  # Dict[str, list[float]]
        self.string_eproperties = {}  # Dict[str, list[str]]
    
    def add_node(self):
        new_node = Vertex(self.N)
        self.N += 1
        #self.vertices.append([])
        self.vertices.append(new_node)
    
    def add_edge(self, src, tgt, srcn, tgtn):
        if VERBOSE:
          print("add",src,"->",tgt)
        node_i = self.vertices[src]
        node_j = self.vertices[tgt]

        new_edge = Edge(node_i.index, node_j.index,srcn,tgtn)
        node_i.edges_source.append(new_edge)
        node_j.edges_target.append(new_edge)
        self.edges.append(new_edge)
        new_edge.index = len(self.edges)-1
        self.M += 1
#        if not self.is_directed:
            #self.vertices[target].append(source)


# fibration_functions.jl:

def initialize(graph):
    N = len(graph.vertices)
    node_labels, unique_labels, components = extract_strong(graph, True)

    if VERBOSE:
      print("number of SCCs",len(components))
    sccs = []
    scc_qty=-1
    nodes_in_sccs_qty=0
    for label in components:
        if len(components[label])>1:
          scc_qty+=1
          #if VERBOSE:
          print("  ",scc_qty,">SCC with more than 1 node, label:",label,"nodes:")
          for c in components[label]:
            print(c+1,end=' ')
          print()
          nodes_in_sccs_qty +=len(components[label])
        new_scc = StrongComponent()
        new_scc.insert_nodes(components[label])
        sccs.append(new_scc)

    partition = [Fiber()]
    autopivot = []
    for strong in sccs:
        strong.check_input(graph)
        strong.classify_strong(graph)
        if strong.type == 0:
            partition[0].insert_nodes(strong.nodes)
        elif strong.type == 1:
            new_fiber = Fiber()
            new_fiber.insert_nodes(strong.nodes)
            partition.append(new_fiber)
        else:
            new_fiber = Fiber()
            new_fiber.insert_nodes(strong.nodes)
            autopivot.append(new_fiber)
            partition[0].insert_nodes(strong.nodes)

    pivot_queue = []
    fiber_index = [-1] * N
    for index, fiber in enumerate(partition):
        pivot_queue.append(fiber.copy())
        for v in fiber.nodes:
            fiber_index[v] = index
        fiber.index = index
    
    for isolated in autopivot:
        pivot_queue.append(isolated.copy())
    
    set_vertices_properties("fiber_index", fiber_index, graph)

    print("nodes in SCCs with more than 1 node:",nodes_in_sccs_qty)
    return partition, pivot_queue

def enqueue_splitted(new_classes, pivot_queue):
    if not new_classes:
        return
    max_fiber = max(new_classes, key=lambda fiber: fiber.number_nodes)
    for fiber in new_classes:
        if fiber is not max_fiber:
            pivot_queue.append(fiber.copy())

def calculate_input_set(graph, pivot, partition, number_edgetype, eprop_name):
    fiber_index = graph.int_vproperties["fiber_index"]
    edgetype_prop = graph.int_eproperties[eprop_name]
    receivers_classes = set()
    input_sets = {}
    for pivot_v in pivot.nodes:
      for edge in graph.vertices[pivot_v].edges_source:
        se = graph.edges[edge.index]
    for pivot_v in pivot.nodes:
        for edge in graph.vertices[pivot_v].edges_source:
            target = edge.target
            etype = edgetype_prop[edge.index]
            if (target) not in input_sets:
                input_sets[target] = [0] * number_edgetype
            input_sets[target][etype-1] += 1
            if (target) < len(fiber_index):
              receivers_classes.add(fiber_index[target])
            else:
              print(f"ERROR: Target {target} is out of bounds (max {len(fiber_index)})")

    receivers = [partition[j].copy() for j in receivers_classes]
    return receivers, input_sets

def split_from_input_set(receivers, input_sets, number_edgetype):
    default_str = "0" * number_edgetype
    final_splitting = {}
    for j, current_fiber in enumerate(receivers):
        sub_input = {}
        for node in current_fiber.nodes:
            inputset_str = "".join(map(str, input_sets.get(node, [0] * number_edgetype)))
            sub_input.setdefault(inputset_str, []).append(node)
        final_splitting[j] = list(sub_input.values())
    return final_splitting

def define_new_partition(final_splitting, receivers, partition, pivot_queue, graph):
    fiber_index = graph.int_vproperties["fiber_index"]
    for j, receiver in enumerate(receivers):
        if len(final_splitting[j]) == 1:
            continue
        partition_fiber = partition[receiver.index]
        new_classes = []
        nodes_to_remove = []
        for arr in final_splitting[j][1:]:
            new_fiber = Fiber()
            new_fiber.index = len(partition)
            new_fiber.insert_nodes(arr)
            for u in new_fiber.nodes:
                fiber_index[u] = new_fiber.index
            partition.append(new_fiber)
            new_classes.append(new_fiber.copy())
            nodes_to_remove.extend(arr)
        partition_fiber.delete_nodes(nodes_to_remove)
        new_classes.append(partition_fiber.copy())
        enqueue_splitted(new_classes, pivot_queue)

def fast_partitioning(graph, pivot, partition, pivot_queue, n_edgetype, eprop_name="edgetype"):
    receivers, input_sets = calculate_input_set(graph, pivot, partition, n_edgetype, eprop_name)
    if VERBOSE:
      print("current grouping of nodes, being refined progressively")
      for p in partition:
        print('[',end='')
        for pn in p.nodes:
          print(pn+1,end=' ')
        print('] ',end='')
      print()
      if len(input_sets)>0:
        print("inputs received by each node")
        for ist in input_sets:
          print("  ",ist+1,end=' [')
          for iii in input_sets[ist]:
            print(iii+1,end=' ')
          print('] ',end='')
        print()
        print("groups that received inputs")
        for r in receivers:
          print('[',end='')
          for rn in r.nodes:
            print(rn+1,end=' ')
          print('] ',end='')
        print()
    final_splitting = split_from_input_set(receivers, input_sets, n_edgetype)
    if VERBOSE:
      print("checks if some nodes should be split into new groups based on the types and amounts of connections they receive")
      for fs in final_splitting:
        for iii in final_splitting[fs]:
          print('[',end='')
          for jjj in iii:
            print(jjj+1,end=' ')
          print('] ',end='')
        print()
      print()
    define_new_partition(final_splitting, receivers, partition, pivot_queue, graph)

def fast_fibration(graph, eprop_name="edgetype"):
    if not graph.is_directed:
        print("Undirected network")
        return
    number_edgetype = len(set(graph.int_eproperties[eprop_name]))
    partition, pivot_queue = initialize(graph)
    while len(pivot_queue)>0:
        pivot_set = pivot_queue.pop(0)
        fast_partitioning(graph, pivot_set, partition, pivot_queue, number_edgetype)
    return partition

def extract_fiber_groups(partition):
    nontrivial_fibers = sum(1 for fiber in partition if fiber.number_nodes > 1)
    total_fibers = len(partition)
    nontriv_qty=0
    for fiber in partition:
      if fiber.number_nodes>1:
        nontriv_qty+=len(fiber.nodes)
    #print("nodes in nontrivial fibers:",nontriv_qty)
    fibers_map = {j: list(fiber.nodes) for j, fiber in enumerate(partition)}
    nontrivial_fibers_list=[]
    trivial_fibers_list=[]
    for fiber in partition:
      if fiber.number_nodes > 1:
        nontrivial_fibers_list.append(list(fiber.nodes))
      elif fiber.number_nodes == 1:
        trivial_fibers_list.append(list(fiber.nodes))
    return nontrivial_fibers, total_fibers, fibers_map

def dump_fiber_map(fm):
  for f in fm:
    for n in fm[f]:
      print(n+1,end=' ')
    print()

# utils.jl
