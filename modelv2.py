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

    
    def forward(self, batch):
        result = []
        for element in batch:
            first_index, second_index = element[0].item(), element[1].item()
            first_graph_internal_encoder = self.internal_graph_encoder.forward(first_index)
            second_graph_internal_encoder = self.internal_graph_encoder.forward(second_index)
            
            first_graph_embedding = self.external_graph_encoder.forward(first_index, first_graph_internal_encoder)
            second_graph_embedding = self.external_graph_encoder.forward(second_index, second_graph_internal_encoder)

            result.append(self.link_prediction_layer.forward(first_graph_embedding, second_graph_embedding))
        to_return =  torch.stack(result)
        reshaped = to_return.view(len(batch), 2)
        return reshaped

class InternalGraphConvolutionLayer(Module):
    def __init__(self):
        super(InternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = 3
        self.W = Parameter(torch.randn(self.node_representation_size,
                                                   self.node_representation_size), requires_grad=True)
        self.M = Parameter(torch.randn(self.node_representation_size,
                                                    self.node_representation_size), requires_grad=True) # globalne
        torch.nn.init.xavier_normal(self.W)
        torch.nn.init.xavier_normal(self.M)
        
    def forward(self, index):  
        self.internal_graph = Graphs.get_internal_graph(index)
        for key, value in self.internal_graph.nodes.items():
            node = value
            sum = torch.mm(node.representation, self.W)
            for adj_vector in node.neighbours:
                    sum += torch.mm(adj_vector.representation, self.M)
            node.representation=F.relu(sum)

        sum = torch.zeros([1, 3], dtype=torch.float32)
        for key,node in self.internal_graph.nodes.items():
            sum = sum + node.representation
        return F.softmax(sum)

class ExternalGraphConvolutionLayer(Module):

    def __init__(self):
        super(ExternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = 3
        self.U = Parameter(torch.randn(self.node_representation_size,
                                        self.node_representation_size), requires_grad=True)
        self.V = Parameter(torch.randn(self.node_representation_size,
                                       self.node_representation_size), requires_grad=True) # globalne
        torch.nn.init.xavier_normal(self.U)
        torch.nn.init.xavier_normal(self.V)

    def forward(self, index, initial_embedding):
        self.node_in_external_graph = Graphs.get_external_graph_node(index)
        self.node_in_external_graph.representation = initial_embedding
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
        self.second_layer = nn.Linear(self.node_representation_size, 2)

    def forward(self, first_node_embedding, second_node_embedding):
        third_tensor = torch.cat((first_node_embedding*second_node_embedding,
                                  first_node_embedding+second_node_embedding), 1)
        ############################
        value = F.relu(self.first_layer(third_tensor))
        x =  F.softmax(self.second_layer(value))
        return x

