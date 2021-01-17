# -*- coding: utf-8 -*-

import sys,random
sys.setrecursionlimit(100000)
import matplotlib.pyplot as pyplot;
import time;
import networkx as nx
import pandas as pd
import pickle
class Graph :
    def __init__ (self) :
        self.edges = {}

    def add_edge (self, from_, to) :
        if from_ not in self.edges :
            self.edges[from_] = [to]
        else :
            self.edges[from_].append(to)

        if to not in self.edges :
            self.edges[to] = []

    def find_reachable_nodes (self, source_nodes) :
        # source nodes is a list of nodes
        reached = set([])
        for node in source_nodes :
            if node in self.edges :
                self.dfs (node, reached)

        return reached

    def dfs (self, node, reached) :
        for nbr in self.edges[node] :
            if nbr not in reached :
                reached.add (nbr)
                self.dfs (nbr, reached)
                


def load_graph(graph_path):
    print("load_graph")
    data = pd.read_csv(graph_path)
    print(data.head())
    edges = data.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def loadFileForGraph(fileName) :
    f = open (fileName)
    lines = f.readlines()

    edges_list = []
    weight_list = []
    nodes_set = set([])
    for line in lines :
        
        if "node_1,node_2" in line  :
            print("inside loop")
            print(line)
            continue
        
        line=line.strip()
       
        node1, node2 = map(int, line.split(','))
        edges_list.append ([node1, node2])
        weight_list.append (random.random()/3)
        # weight_list.append (1)
        nodes_set.add(node1)
        nodes_set.add(node2)

    return edges_list, weight_list, nodes_set

def generateRamdomGraphs (num_graphs, edges, probs) :
    graph_snapshots = []
    for i in range (num_graphs) :
        tmp_graph = Graph ()
        for edge, prob in zip(edges, probs) : 
            rand = random.random()
            if rand < prob :
                tmp_graph.add_edge (edge[0], edge[1])
        graph_snapshots.append(tmp_graph)

    return graph_snapshots

def getInfluenceMap(nodes,graphSnaps, threshold) :
    print("getInfluenceMap: ")
    influencedMap = {};
    for node in nodes:
        node_set = set();
        node_set.add(node);
        influencedMap[node] = findInfluence(node_set,graphSnaps,threshold);
    return influencedMap;

def findInfluence (nodes, graphSnaps, threshold) :
    influenced_node_count = {} # this contains nodes and count for each node
    for G in graphSnaps :
        S = G.find_reachable_nodes (nodes)
        # Update influenced_node_count looking at S
        for node in S :
            if node in influenced_node_count :
                influenced_node_count[node]+=1
            else :
                influenced_node_count[node] = 1

    influenced_nodes = set([])
    for node in influenced_node_count :
        if influenced_node_count [node] > threshold:
            influenced_nodes.add (node)

    # We do not want source nodes in the influenced set. Remove any nodes that are there in source nodes from influenced_nodes.
    for node in nodes :
        if node in influenced_nodes :
            influenced_nodes.remove (node)

    return influenced_nodes

def heuristic1 (graph_snaps, nodes_set, k, step_size, threshold,influenceMap,selectedSet) :
    load_influence_map_from_file = 0
    # graph_snaps: graph snapshots
    # nodes_set: set of nodes in the complete graph
    # k: number of nodes to influence initially
    # step_size: number of nodes to add to the opt set every iteration
    uninfluencedNodes = nodes_set;


    bestNodes = set();
    maxLength = 0;
    maxNode = -1;
    for node in selectedSet:
        try:
            bestNodes = set.union(influenceMap[node],bestNodes);
        except:
            print("An exception occurred")
        bestNodes.add(node);
        uninfluencedNodes.discard(node);

        uninfluencedNodes = uninfluencedNodes.difference(bestNodes);


    if not load_influence_map_from_file :
        f = open ("influenceMapObject.pickle", "wb")
        pickle.dump (influenceMap, f)
        f.close ()
    else :
        f = open ("influenceMapObject.pickle", "rb")
        influenceMap = pickle.load (f)


    for uninfluenced_node in uninfluencedNodes:

        new_nodes_influenced = len(set.intersection(influenceMap[uninfluenced_node], uninfluencedNodes));
        if maxLength < new_nodes_influenced:
            maxLength = new_nodes_influenced;
            maxNode = uninfluenced_node;

        # print ("best nodes before", len(bestNodes))

    bestNodes.add(maxNode);
    try:
        bestNodes = set.union(bestNodes,influenceMap[maxNode]);
    except:
         print("An exception occurred")
    selectedSet.add(maxNode);

    # print ("uninfluenced nodes", len(uninfluencedNodes))

    return  bestNodes;

def getNodesThatConnectMaxNodes(graph_path):
    print("load_graph")
    data = pd.read_csv(graph_path)
    print(data.columns)
    nodesConnectedToMorenetwork = data[['node_1','node_2']].groupby(['node_1'])['node_1'].size().nlargest(10).reset_index(name='top10')
    nodesConnectedToMoreNt = nodesConnectedToMorenetwork['node_1'].tolist()
    print(nodesConnectedToMoreNt)
    return nodesConnectedToMoreNt

def main(filePath,selectedNodes,threshold,noOfRamdomGraph,pngFileName,plotNo,color,marker):
    print("GALAssignment",threshold)
    #graph=load_graph(filePath)
    #print(graph.edges)
    edgesDetails, weightProp, nodes =loadFileForGraph(filePath)
    graphSnaps = generateRamdomGraphs (noOfRamdomGraph, edgesDetails, weightProp)
    influencedMap=getInfluenceMap(nodes,graphSnaps,threshold)
    step_size=1
    ltModelSelectedSet = set()
    ltModelHeuristicTime = []
    ltModelHeuristic = []
    for k in selectedNodes:
        print ("k = ", k)

        if k == 0:
            continue;
        
        startTime = time.time();
        influenceSet = heuristic1(graphSnaps, nodes, k, step_size, threshold,influencedMap,ltModelSelectedSet);
        ltModelHeuristicTime.append(time.time() - startTime);
        ltModelHeuristicCount = len(influenceSet)
        ltModelHeuristic.append(ltModelHeuristicCount)
        #print(influenceSet)
        print("Size of influenced set is ", len(influenceSet));
        print("Selected set for LT MODEL is : ",ltModelSelectedSet)
    xi = list(sorted(selectedNodes))
    pyplot.subplot(plotNo)
    pyplot.plot(xi, ltModelHeuristicTime, color, label='Linear Thershold model', marker=marker,linestyle='--')
    pyplot.legend(loc='upper left')
    pyplot.ylabel('Total execution time');
    pyplot.xlabel('nodes selected');
    pyplot.draw()
    pyplot.savefig(pngFileName);
            

if __name__ == "__main__":
    filePath="lastfm_asia_edges.csv"
    selectedNodes={1,6,7,10,21,22,32,34,40,80}
    threshold=10
    noOfRamdomGraph=50
    pngFileName='LT_MODEL_GIVEN_NODES'
    plotNo=111
    color='-b'
    marker='o'
    main(filePath,selectedNodes,threshold,noOfRamdomGraph,pngFileName,plotNo,color,marker)
    plotNo=222
    color='-r'
    marker='*'
    selectedNodes=set(getNodesThatConnectMaxNodes(filePath))
    pngFileName='LT_MODEL_MAX_CONNECTED_NODES'
    main(filePath,selectedNodes,threshold,noOfRamdomGraph,pngFileName,plotNo,color,marker)