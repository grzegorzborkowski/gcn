from Graphs import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch


class DCNNv2(nn.Module):
    def __init__(self):
        super(DCNNv2, self).__init__()

        self.internal_graph_encoder = InternalGraphConvolutionLayer()

        self.external_graph_encoder = ExternalGraphConvolutionLayer()

        self.link_prediction_layer = LinkPredictionLayer()

    
    def forward(self, first_index, second_index):
        first_graph_internal_encoder = self.internal_graph_encoder.forward(first_index)
        second_graph_internal_encoder = self.internal_graph_encoder.forward(second_index)
        
        first_graph_embedding = self.external_graph_encoder.forward(first_index, first_graph_internal_encoder)
        second_graph_embedding = self.external_graph_encoder.forward(second_index, second_graph_internal_encoder)

        return self.link_prediction_layer.forward(first_graph_embedding, second_graph_embedding)

class InternalGraphConvolutionLayer(Module):
    def __init__(self):
        super(InternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = 3
        self.W = Parameter(torch.randn(self.node_representation_size,
                                                   self.node_representation_size))
        self.M = Parameter(torch.randn(self.node_representation_size,
                                                    self.node_representation_size)) # globalne
        
    def forward(self, index):
        self.internal_graph = Graphs.get_internal_graph(index)
        for node in self.internal_graph.nodes:
            sum = torch.mm(node.representation, self.W)
            for adj_vector in node.neighbours:
                    sum += torch.mm(adj_vector.representation, self.M)
            node.representation=F.relu(sum)

        sum = torch.zeros([1, 3], dtype=torch.float32)
        for node in self.internal_graph.nodes:
            sum = sum + node.representation
        return F.softmax(sum)

class ExternalGraphConvolutionLayer(Module):

    def __init__(self):
        super(ExternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = 3
        self.U = Parameter(torch.randn(self.node_representation_size,
                                        self.node_representation_size))
        self.V = Parameter(torch.randn(self.node_representation_size,
                                       self.node_representation_size)) # globalne

    def forward(self, index, initial_embedding):
        self.node_in_external_graph = Graphs.get_external_graph_node(index)
        ######################## USE INITIAL EMBEDDING FRM INTERNAL GRAPH ########################
        for node in self.node_in_external_graph.neighbours:
            sum = torch.mm(node.representation, self.U)
            for adj_vector in node.neighbours:
                    sum += torch.mm(adj_vector.representation, self.V)
            node.representation=F.relu(sum)
        return F.softmax(node.representation)

class LinkPredictionLayer(Module):

    def __init__(self):
        super(LinkPredictionLayer, self).__init__()
        self.node_representation_size = 3
        self.first_layer = nn.Linear(self.node_representation_size*2, self.node_representation_size)
        self.second_layer = nn.Linear(self.node_representation_size, 1)

    def forward(self, first_node_embedding, second_node_embedding):
        third_tensor = torch.cat((first_node_embedding*second_node_embedding,
                                  first_node_embedding+second_node_embedding), 1)
        ############################
        value = F.relu(self.first_layer(third_tensor))
        return F.relu(self.second_layer(value))

