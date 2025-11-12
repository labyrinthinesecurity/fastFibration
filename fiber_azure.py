#!/usr/bin/python3
from azure_fast_fibration import *
import json
import ast
import subprocess
import csv,os
import math,hashlib
import argparse
import pandas as pd
import random
import uuid
import numpy as np
import sys
from datetime import datetime

current_date = datetime.now()
timestamp = current_date.strftime("%Y-%m-%d")


COLORS = ["#435055", "#7694a1", "#a5c3ab", "#ffa500", "#ffde21"]

spnscache={}
df=None

warpermdict={}
spnscache={}
spn={}
fiber={}
pid_dict={}
rdid_dict={}
group={}

pid_dict = {}
rdid_dict = {}
reverse_pid_dict = {}
reverse_rdid_dict = {}

def hash_rdids(rdid_list):
    # Sort to ensure consistent hash regardless of order
    rdid_str = ','.join(sorted(map(str, rdid_list)))
    return hashlib.sha256(rdid_str.encode()).hexdigest()


def generate_AZGRAPH():
  if os.path.exists(f'sorted_NHIs_{timestamp}.csv'):
    score_df = pd.read_csv(f'sorted_NHIs_{timestamp}.csv')
  else:
    print(f"ERROR, cannot read sorted_NHIs_{timestamp}.csv ====> please run NHIs.py first")
    sys.exit()
  score_filtered = score_df[['pid', 'name', 'WAR', 'blast_radius']]
  if os.path.exists('AZURE_FRS.csv'):
    df = pd.read_csv('AZURE_FRS.csv', usecols=['pid', 'rdid'])
  else:
    print("ERROR, cannot read AZURE_FRS.csv ====> please run generate_AZRBAC()")
    sys.exit()
  c=0
  for p in df['pid']:
    if p not in pid_dict:
      c+=1
      pid_dict[p]=c

  for p in df['rdid']:
    if p not in rdid_dict:
      c+=1
      rdid_dict[p]=c

  reverse_pid_dict = {v: k for k, v in pid_dict.items()}
  reverse_rdid_dict = {v: k for k, v in rdid_dict.items()}

  # Assign unique integer values to each unique pid and rdid
  df['pid_unique'] = df['pid'].apply(lambda x: pid_dict[x])  # Correct way to map
  df['rdid_unique'] = df['rdid'].apply(lambda x: rdid_dict[x])  # Same for rdid
  df['edgetype'] = 'plus'

  df[['rdid','pid','edgetype','rdid_unique', 'pid_unique']].to_csv("AZGRAPH.csv", index=False, header=['SourceName','TargetName','Type','Source','Target'])

def find_node(nodeuuid):
    for key, value in nodenamescache.items():
        if value == nodeuuid:
            return key
    return None

def Azure_extract_fiber_groups(partition):
    nontrivial_fibers = sum(1 for fiber in partition if fiber.number_nodes > 1)
    total_fibers = len(partition)
    nontriv_qty=0
    for fiber in partition:
      if fiber.number_nodes>1:
        nontriv_qty+=len(fiber.nodes)
    print("nodes in nontrivial fibers:",nontriv_qty)
    fibers_map = {j: list(fiber.nodes) for j, fiber in enumerate(partition)}
    nontrivial_fibers_list=[]
    trivial_fibers_list=[]
    user_fibers=[]
    for fiber in partition:
      if fiber.number_nodes > 1:
        nontrivial_fibers_list.append(list(fiber.nodes))
      elif fiber.number_nodes == 1:
        trivial_fibers_list.append(list(fiber.nodes))
    #print("Non-trivial User fibers:")
    for i,nf in enumerate(nontrivial_fibers_list):
      uf={}
      for an in nf:
        if nodenamescache[str(an+1)] in spnscache:
          if 'representative' not in uf:
            uf['representative']=an
            uf['orbit']=[]
          uf['orbit'].append(nodenamescache[str(an+1)])
      if len(uf)>0:
        user_fibers.append(uf)
    for i,nf in enumerate(trivial_fibers_list):
      uf={}
      for an in nf:
        if ':' not in nodenamescache[str(an+1)]:   # we filter to get only user fibers, not other fibers
          uf['representative']=an
          uf['orbit']=[]
          uf['orbit'].append(nodenamescache[str(an+1)])
      if len(uf)>0:
        user_fibers.append(uf)
    return nontrivial_fibers, total_fibers, fibers_map,user_fibers

def generate_base(user_fibers,collapsed=True):
  nodes={}
  colors={}
  edges = []
  rep_nodes = {}
  fiber_nodes = {}
  fiber=0
  for uf in user_fibers:
    fiber+=1
    ancs=[]
    rep=''
    for u in uf['orbit']:
      rep=uf['representative']+1
      if rep not in rep_nodes:
        rep_nodes[nodenamescache[str(uf['representative']+1)]]={}
        rep_nodes[nodenamescache[str(uf['representative']+1)]]['id']=fiber
        if nodenamescache[str(uf['representative']+1)] in spnscache:
          rep_nodes[nodenamescache[str(uf['representative']+1)]]['name']=spnscache[nodenamescache[str(uf['representative']+1)]]['displayName']
        else:
          rep_nodes[nodenamescache[str(uf['representative']+1)]]['name']="root"
        rep_nodes[nodenamescache[str(uf['representative']+1)]]['rdids']= df[df['TargetName'] == nodenamescache[str(uf['representative']+1)]]['SourceName'].tolist()
      n=find_node(u)
      if n is None:
        print("weird..")
        continue
      if uf['representative']+1!=n:
        if collapsed==False:
          ancs.append(int(n))
    fiber_nodes[nodenamescache[str(uf['representative']+1)]]=ancs
  nodes['root']='root'
  for u in rep_nodes:
    nodes[u] = u
    if u=='root':
      colors[u] = 'black'
    else:
      colors[u] = COLORS[4]
    edges.append(('root',u))
    for n in rep_nodes[u]['rdids']:
        key=str(fiber)+"_"+n
        nodes[key] = n
        colors[key] = COLORS[1]
        edges.append((u,key))
    for v in fiber_nodes[u]:
      if v not in nodes:
        nodes[v] = v
        colors[v] = COLORS[3]
        edges.append((v,u))
  for edge in edges:
    if edge[0] in nodes and edge[1] in nodes:
      pass
    else:
      print("ISSUE",edge[0],"->",edge[1])
      sys.exit()
  graph = Network(notebook=True, directed=True, cdn_resources='in_line',height='1000px')
  for node in nodes:
    graph.add_node(node,label=nodes[node],physics=True,color=colors[node])
  for edge in edges:
    graph.add_edge(edge[0], edge[1], physics=False)
  if collapsed:
    graph.show(f"Azure_fibers.html")
  else:
    graph.show(f"Azure_orbits.html")


if os.path.exists("spns_cache.json"):
  with open("spns_cache.json", "r") as file:
    spnscache=json.load(file)
else:
  print("ERROR: please load SPNs from Entra by running gneerate_spns_cache first")
  sys.exit()

#generate_AZGRAPH()
graph,df = graph_from_csv("AZGRAPH.csv", is_directed=True)

print("n=",graph.N)
dump_fibration_graph_to_csv(graph,"graph_map.csv")

fiber_partition = fast_fibration(graph)
number_nontrivial_fibers, total_fibers, fiber_map, user_fibers = Azure_extract_fiber_groups(fiber_partition)

nontriv={}

cnt=-1
for f in fiber_map:
  if len(fiber_map[f])>1:
    cnt+=1
    nontriv[f]=fiber_map[f]
print("non trivial fibers:",cnt)
dump_fibration_base_to_csv(graph,nontriv,fiber_map,"fiber_map.csv")

print("number of fibers containing more than one node (nontrivial)",number_nontrivial_fibers)
print("total number of fibers (including single node fibers)",total_fibers)
for i,u in enumerate(user_fibers):
#  print("FIBER",i,u['representative']+1,nodenamescache[str(u['representative']+1)],len(u['orbit']))
  print("FIBER",i,"NHIs population count",len(u['orbit']))
  print("  NHIs in fiber:",u['orbit'])
