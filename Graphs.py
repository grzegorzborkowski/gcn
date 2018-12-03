import csv
import torch
import collections
import abc
import os
import numpy as np
import sklearn.model_selection

class Graphs:

    node_representation_size = 3
    external_graph = None
    internal_graphs = {} # id_of_node_in_external_graph -> internal_graph
    train_to_valid_ratio = 0.7
    unique_internal_nodes = 2300
    
    @staticmethod
    def initialize():
        Graphs.initialize_external_graph()
        Graphs.intialize_internal_graphs()

    @staticmethod
    def initialize_external_graph():
        Graphs.external_graph = ExternalGraph()
        with open("../dataset/external_graph.csv") as external_graph_file:
            csv_reader = csv.reader(external_graph_file)
            for row_list in csv_reader:
                row_list = [row.strip() for row in row_list]
                node = Graphs.external_graph.get_or_create_node_external(int(row_list[0]))
                for x in row_list[1:]:
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

    @staticmethod
    def initialize_single_internal_graph_from_file(file_path, id):
        if id in Graphs.internal_graphs:
            raise ValueError("Redundant internal graph" + str(id))
        else:
            Graphs.internal_graphs[id] = InternalGraph()

        with open(file_path) as internal_graph_file:
            csv_reader = csv.reader(internal_graph_file)
            for row_list in csv_reader:
                row_list = [row.strip() for row in row_list]
                for element in range(1, len(row_list)):
                    if str(element) != "None":
                        #print(str(id) + " " + str(element) + " " + str(row_list[element]))
                        first_node = Graphs.internal_graphs[id].get_or_create_node_internal(int(row_list[0]))
                        second_node = Graphs.internal_graphs[id].get_or_create_node_internal(int(row_list[element]))
                        first_node.add_neighbour(second_node)
                
    @staticmethod
    def get_internal_graph(index):
        if index in Graphs.internal_graphs:
            return Graphs.internal_graphs[index]
        else:
            raise ValueError("No such node in internal graphs " + str(index))
    
    @staticmethod
    def get_external_graph_node(index):
        return Graphs.external_graph.nodes[index]

    @staticmethod
    def get_train_valid_examples():
        samples_X, samples_Y = [], []
        for key, value in Graphs.external_graph.nodes.items():
            for neighbour in value.neighbours:
                samples_X.append([key, neighbour.id])
                samples_Y.append([0,1]) # like one-hot encoding
        samples_X, smaples_Y = np.array(samples_X), np.array(samples_Y)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(samples_X, samples_Y)
        print (len(X_train))
        return torch.from_numpy(X_train), torch.from_numpy(X_test), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

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
    
    def get_or_create_node_internal(self, id):
        if id in self.nodes:
            return self.nodes[id]
        else:
            node = Node(id)
            self.nodes[id] = node
            return self.nodes[id]

class Node():

    def __init__(self, id, representation=torch.randn(1, Graphs.node_representation_size)):
        self.representation = representation
        self.id = id
        self.neighbours = []
        #torch.nn.init.xavier_normal(self.representation)

    def add_neighbour(self, node):
        node_exist = False
        node_exists_in_neighbour = False
        for neighbour in self.neighbours:
            if neighbour.id == node.id:
                node_exist = True
                break
        if not node_exist:
            self.neighbours.append(node)
        
        for neighbour in node.neighbours:
            if self.id == neighbour.id:
                node_exists_in_neighbour = True
                break
        if not node_exists_in_neighbour:
            node.neighbours.append(self)

    def __str__(self):
        return "Node: " + str(self.id) + " " +  str(len(self.neighbours)) #str(self.representation) + " : " + str(len(self.neighbours))