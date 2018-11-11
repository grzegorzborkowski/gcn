from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch

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
    def __init__(self, internal_graph, node_representation_size):
        super(InternalGraphConvolutionLayer, self).__init__()
        self.internal_graph = internal_graph
        self.W = Parameter(torch.randn(node_representation_size,
                                                   node_representation_size))
        self.M = Parameter(torch.randn(node_representation_size,
                                                    node_representation_size)) # globalne
        self.node_vectors = [] # init properly should be a matrix
        self.T = 1

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


#class InternalRepresentationLayer(Module):
#    def __init__(self, vectors_for_nodes):
#        super(InternalGraphConvolutionLayer, self).__init__()
#        self.vectors_for_nodes = vectors_for_nodes

#    def forward(self, input, adj):
#        return softmax(dodaj_wektorowo_wierzcholki)


#class ExternalGraphConvolutionLayer(Module):
#
#    def __init__(self):
#        pass

