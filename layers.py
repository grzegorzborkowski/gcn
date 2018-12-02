from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import Graphs

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

    def __init__(self, representation):
        self.representation = representation
        self.neighbours = []

    def add_neighbour(self, node):
        self.neighbours.append(node)
        node.neighbours.append(self)

    def __str__(self):
        return "Node: " + str(self.representation)

class InternalGraphConvolutionLayer(Module):
    def __init__(self, internal_graph, node_representation_size=Graphs.Graphs.node_representation_size):
        super(InternalGraphConvolutionLayer, self).__init__()
        self.internal_graph = internal_graph
        self.W = Parameter(torch.randn(node_representation_size,
                                                   node_representation_size))
        self.M = Parameter(torch.randn(node_representation_size,
                                                    node_representation_size)) # globalne

    def forward(self):
        #print (self.W)
        #print (self.M)
        #print ("before", self.internal_graph)
        for node in self.internal_graph.nodes:
            sum = torch.mm(node.representation, self.W)
            for adj_vector in node.neighbours:
                    sum += torch.mm(adj_vector.representation, self.M)
            node.representation=F.relu(sum)
        #print ("after", self.internal_graph)


class InternalRepresentationLayer(Module):
    def __init__(self, internal_graph = None):
        super(InternalRepresentationLayer, self).__init__()
        self.internal_graph = internal_graph

    def forward(self):
        sum = torch.zeros([1, Graphs.Graphs.node_representation_size], dtype=torch.float32)
        for node in self.internal_graph.nodes:
            sum = sum + node.representation
        return F.softmax(sum)

class ExternalGraphConvolutionLayer(Module):

    def __init__(self, node_in_external_graph, node_representation_size=Graphs.Graphs.node_representation_size):
        super(ExternalGraphConvolutionLayer, self).__init__()
        self.node_in_external_graph = node_in_external_graph
        self.U = Parameter(torch.randn(node_representation_size,
                                        node_representation_size))
        self.V = Parameter(torch.randn(node_representation_size,
                                       node_representation_size)) # globalne

    def forward(self):
        for node in self.node_in_external_graph.neighbours:
            sum = torch.mm(node.representation, self.U)
            for adj_vector in node.neighbours:
                    sum += torch.mm(adj_vector.representation, self.V)
            node.representation=F.relu(sum)
        return F.softmax(node.representation)

class LinkPredictionLayer(Module):

    def __init__(self, external_graph, node_representation_size=Graphs.Graphs.node_representation_size):
        super(LinkPredictionLayer, self).__init__()
        self.first_layer = nn.Linear(node_representation_size*2, node_representation_size)
        self.second_layer = nn.Linear(node_representation_size, 1)
        self.external_graph = external_graph

    def forward(self, first_idx, second_idx):
        first_node_representation = self.external_graph.nodes[first_idx].representation
        second_node_representation = self.external_graph.nodes[second_idx].representation
        third_tensor = torch.cat((first_node_representation*second_node_representation,
                                  first_node_representation+second_node_representation), 1)
        ############################
        value = F.relu(self.first_layer(third_tensor))
        return F.relu(self.second_layer(value))