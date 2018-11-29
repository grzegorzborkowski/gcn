import csv
import torch
import collections

class Graphs:

    node_representation_size = 3
    external_graph = None
    
    @staticmethod
    def initialize():
        Graphs.external_graph = ExternalGraph()
        with open("../dataset/external_graph_2.csv") as external_graph_file:
            csv_reader = csv.reader(external_graph_file)
            for row_list in csv_reader:
                row_list = [row.strip() for row in row_list]
                node = Graphs.external_graph.get_or_create_node_external(int(row_list[0]))
                for x in row_list[1:]:
                    if x != "":
                        node.add_neighbour(Graphs.external_graph.get_or_create_node_external(int(x)))

    @staticmethod
    def get_internal_graph(index):
        return None
    
    @staticmethod
    def get_external_graph_node(index):
        return self.external_graph[index]

    @staticmethod
    def get_train_valid_examples():
        pass

class ExternalGraph():

    def __init__(self):
        self.nodes = {} # id -> Node

    def get_or_create_node_external(self, id):
        if id in self.nodes:
            return self.nodes[id]
        else:
            node = Node(id)
            self.nodes[id] = node
            return self.nodes[id]

    def print_external_graph(self):
        for node in self.nodes.values():
            print (node)

class Graph():

    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def __str__(self):
        result = ""
        for node in self.nodes:
            result += str(node) + "\n"
        return result

class Node():

    def __init__(self, id, representation=torch.randn(Graphs.node_representation_size, 1)):
        self.representation = representation
        self.id = id
        self.neighbours = []

    def add_neighbour(self, node):
        self.neighbours.append(node)
        node.neighbours.append(self)

    def __str__(self):
        return "Node: " + str(self.id) + " " +  str(len(self.neighbours)) #str(self.representation) + " : " + str(len(self.neighbours))