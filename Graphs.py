import csv
import torch
import collections
import abc
import os

class Graphs:

    node_representation_size = 3
    external_graph = None
    internal_graphs = {} # id_of_node_in_external_graph -> internal_graph
    
    @staticmethod
    def initialize():
        Graphs.initialize_external_graph()
        Graphs.intialize_internal_graphs()

    @staticmethod
    def initialize_external_graph():
        Graphs.external_graph = ()
        with open("../dataset/external_graph_2.csv") as external_graph_file:
            csv_reader = csv.reader(external_graph_file)
            for row_list in csv_reader:
                row_list = [row.strip() for row in row_list]
                node = Graphs.external_graph.get_or_create_node_external(int(row_list[0]))
                for x in row_list[ExternalGraph1:]:
                    if x != "":
                        node.add_neighbour(Graphs.external_graph.get_or_create_node_external(int(x)))

    @staticmethod
    def intialize_internal_graphs():
        path = "../dataset/internal_graphs/"
        directory_with_graphs = os.listdir(path)
        for graph in directory_with_graphs:
            id = int(graph.split(".")[0])
            full_path = path + graph
            Graphs.initialize_single_internal_graph_from_file(full_path, id)
            
            # with open(full_path) as internal_graph_file:
            #     csv_reader = csv.reader(internal_graph_file)
            #     for row_list in csv_reader:
            #         row_list = [row.strip() for row in row_list]
            #         print (row_list)

    @staticmethod
    def initialize_single_internal_graph_from_file(file_path, id):
        if id in Graphs.internal_graphs:
            raise ValueError("Redundant internal graph")
        else:
            Graphs.internal_graphs[id] = InternalGraph()
            
        with open(file_path) as internal_graph_file:
            csv_reader = csv.reader(internal_graph_file)


    @staticmethod
    def get_internal_graph(index):
        return None
    
    @staticmethod
    def get_external_graph_node(index):
        return self.external_graph[index]

    @staticmethod
    def get_train_valid_examples():
        pass

class AbstractGraph(abc.ABC):
    def __init__(self):
        self.nodes = {} # id -> Node

    def get_or_create_node_external(self, id):
        if id in self.nodes:
            return self.nodes[id]
        else:
            node = Node(id)
            self.nodes[id] = node
            return self.nodes[id]

    def print_graph(self):
        for node in self.nodes.values():
            print (node)

class ExternalGraph(AbstractGraph):
    pass

class InternalGraph(AbstractGraph):
    pass

    

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