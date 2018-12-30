from Graphs import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DCNNv2(nn.Module):

    def __init__(self):
        super(DCNNv2, self).__init__()

        self.internal_graph_encoder = InternalGraphConvolutionLayer()

        self.external_graph_encoder = ExternalGraphConvolutionLayer()

        self.link_prediction_layer = LinkPredictionLayer()

    
    def forward(self, batch):
        result = []
        for idx, element in enumerate(batch):
            first_index, second_index = element[0].item(), element[1].item()
            first_graph_internal_encoder = self.internal_graph_encoder.forward(first_index)
            second_graph_internal_encoder = self.internal_graph_encoder.forward(second_index)
            
            external_graph = Graphs.external_graph
            first_graph_neighbours = external_graph.nodes[first_index].neighbours
            list_of_first_graph_neighbours_embedding = []
            for node in first_graph_neighbours:
                list_of_first_graph_neighbours_embedding.append(self.internal_graph_encoder.forward(node.id))
            
            second_graph_neighbours = external_graph.nodes[second_index].neighbours
            list_of_second_graph_neighbour_embedding = []
            for node in second_graph_neighbours:
                list_of_second_graph_neighbour_embedding.append(self.internal_graph_encoder.forward(node.id))

            first_graph_embedding = self.external_graph_encoder.forward(first_index, first_graph_internal_encoder, list_of_first_graph_neighbours_embedding)
            second_graph_embedding = self.external_graph_encoder.forward(second_index, second_graph_internal_encoder, list_of_second_graph_neighbour_embedding)

            result.append(self.link_prediction_layer.forward(first_graph_embedding, second_graph_embedding))
        to_return = torch.stack(result)
        reshaped = to_return.view(len(batch), 2)
        return reshaped

    # TODO: refactor function above by reusing this one
    def get_node_embedding(self, idx):
        internal_encoding = self.internal_graph_encoder.forward(idx)
        external_graph = Graphs.external_graph
        first_graph_neighbours = external_graph.nodes[idx].neighbours
        list_of_first_graph_neighbours_embedding = []
        for node in first_graph_neighbours:
            list_of_first_graph_neighbours_embedding.append(self.internal_graph_encoder.forward(node.id))
        first_graph_embedding = self.external_graph_encoder.forward(idx, internal_encoding, list_of_first_graph_neighbours_embedding)
        return first_graph_embedding
            

class InternalGraphConvolutionLayer(Module):
    def __init__(self):
        super(InternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = Graphs.node_representation_size
        self.how_many_internal_nodes_type = Graphs.unique_internal_nodes
        self.W = Parameter(torch.randn(self.node_representation_size,
                                                   self.node_representation_size), requires_grad=True)
        self.M = Parameter(torch.randn(self.node_representation_size,
                                                    self.node_representation_size), requires_grad=True) 
        self.Internal_Node_impact = nn.Embedding(self.how_many_internal_nodes_type, self.node_representation_size, sparse=False)
        torch.nn.init.xavier_normal_(self.W)
        torch.nn.init.xavier_normal_(self.M)

    def visualize_internal_nodes_embedding(self):
        all_samples = []
        for idx in range(0, self.how_many_internal_nodes_type):
            all_samples.append(self.Internal_Node_impact(torch.LongTensor([idx])).detach().numpy().flatten())
        X_embedded = TSNE(n_components=2).fit_transform(all_samples)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
        plt.show()

        
    def forward(self, index):
        result = 0
        sum = torch.randn(self.node_representation_size, 1)
        self.internal_graph = Graphs.get_internal_graph(index)
        for key, value in self.internal_graph.nodes.items():
            graph_value = self.Internal_Node_impact(torch.LongTensor([key]))
            sum = torch.mm(self.W, graph_value.reshape(self.node_representation_size, 1))
            for adj_vector in self.internal_graph.nodes[key].neighbours:
                vec = self.Internal_Node_impact(torch.LongTensor([adj_vector.id]))
                vec = vec.reshape(self.node_representation_size,1)
                sum += torch.mm(self.M, vec)
            sum = F.relu(sum)
            result += sum
        to_return =  F.softmax(result, dim=0)
        return to_return

class ExternalGraphConvolutionLayer(Module):

    def __init__(self):
        super(ExternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = Graphs.node_representation_size
        self.U = Parameter(torch.randn(self.node_representation_size,
                                        self.node_representation_size), requires_grad=True)
        self.V = Parameter(torch.randn(self.node_representation_size,
                                       self.node_representation_size), requires_grad=True) # globalne
        torch.nn.init.xavier_normal_(self.U)
        torch.nn.init.xavier_normal_(self.V)

    def forward(self, index, initial_embedding, neighbours_embeddings):
        result = torch.mm(self.U, initial_embedding)
        for neighbour in neighbours_embeddings:
           result += torch.mm(self.V, neighbour)
        result = F.relu(result)
        to_return = F.softmax(result, dim=0)
        return to_return

class LinkPredictionLayer(Module):

    def __init__(self):
        super(LinkPredictionLayer, self).__init__()
        self.node_representation_size = Graphs.node_representation_size
        self.first_layer = nn.Linear(self.node_representation_size*2, self.node_representation_size)
        self.second_layer = nn.Linear(self.node_representation_size, 2)
    def forward(self, first_node_embedding, second_node_embedding):
        third_tensor = torch.cat((first_node_embedding, second_node_embedding), 0)
        third_tensor = third_tensor.reshape(1,Graphs.node_representation_size*2)
        x = F.leaky_relu(self.first_layer(third_tensor))
        x = self.second_layer(x)
        x = F.softmax(x, dim=1)
        return x