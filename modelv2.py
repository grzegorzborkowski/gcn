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
        
        # print("DCNNv2 result " + str(result))
        # print (result)
        to_return = torch.stack(result)
        # print("DCNNv2 to return " + str(to_return))
        reshaped = to_return.view(len(batch), 2)
        return reshaped

class InternalGraphConvolutionLayer(Module):
    def __init__(self):
        super(InternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = Graphs.node_representation_size
        self.how_many_internal_nodes_type = Graphs.unique_internal_nodes
        self.W = Parameter(torch.randn(self.node_representation_size,
                                                   self.node_representation_size), requires_grad=True)
        self.M = Parameter(torch.randn(self.node_representation_size,
                                                    self.node_representation_size), requires_grad=True) # globalne
        self.Internal_Node_Impact = Parameter(torch.randn(self.how_many_internal_nodes_type, self.node_representation_size), requires_grad=True)
        torch.nn.init.xavier_normal(self.W)
        torch.nn.init.xavier_normal(self.M)
        torch.nn.init.xavier_normal(self.Internal_Node_Impact)
        
    def forward(self, index):
        result = 0
        self.internal_graph = Graphs.get_internal_graph(index)
        for key, value in self.internal_graph.nodes.items():
            graph_value = self.Internal_Node_Impact[key, :]
            sum = torch.mm(self.W, graph_value.reshape(self.node_representation_size, 1))
            for adj_vector in self.internal_graph.nodes[key].neighbours:
                    vec = self.Internal_Node_Impact[adj_vector.id, :]
                    vec = vec.reshape(self.node_representation_size,1)
                    sum += torch.mm(self.M, vec)
            #node.representation=F.relu(sum)
            sum = F.relu(sum)
            #print (sum)
            result += sum

        #sum = torch.zeros([1, 3], dtype=torch.float32)
        #for key,node in self.internal_graph.nodes.items():
        #    sum = sum + node.representation
        return F.softmax(result)

class ExternalGraphConvolutionLayer(Module):

    def __init__(self):
        super(ExternalGraphConvolutionLayer, self).__init__()
        self.node_representation_size = Graphs.node_representation_size
        self.U = Parameter(torch.randn(self.node_representation_size,
                                        self.node_representation_size), requires_grad=True)
        self.V = Parameter(torch.randn(self.node_representation_size,
                                       self.node_representation_size), requires_grad=True) # globalne
        torch.nn.init.xavier_normal(self.U)
        torch.nn.init.xavier_normal(self.V)

    def forward(self, index, initial_embedding, neighbours_embeddings):
        result = initial_embedding
        result = torch.mm(self.U, initial_embedding)
        for neighbour in neighbours_embeddings:
            result += torch.mm(self.V, neighbour)
        result = F.relu(result)
        result = F.softmax(result)
        #print (result)
        #self.node_in_external_graph.representation = result
        return result

        # ######################## USE INITIAL EMBEDDING FRM INTERNAL GRAPH ########################
        # for node in self.node_in_external_graph.neighbours:
        #     sum = torch.mm(node.representation, self.U)
        #     for adj_vector in node.neighbours:
        #         print(str(torch.mm(adj_vector.representation, self.V)))
        #         sum += torch.mm(adj_vector.representation, self.V)
        #     node.representation=F.relu(sum)
        # print(str(F.softmax(sum)))
        # to_return = F.softmax(node.representation)
        # node.representation
        # return F.softmax(sum)

class LinkPredictionLayer(Module):

    def __init__(self):
        super(LinkPredictionLayer, self).__init__()
        self.node_representation_size = Graphs.node_representation_size
        print (self.node_representation_size)
        self.first_layer = nn.Linear(self.node_representation_size*2, self.node_representation_size)
        self.second_layer = nn.Linear(self.node_representation_size, 2)

    def forward(self, first_node_embedding, second_node_embedding):
        return F.softmax(torch.randn(2,1))
        # print("fne", first_node_embedding)
        #print ("sne", second_node_embedding)
        third_tensor = torch.cat((first_node_embedding*second_node_embedding,
                                  first_node_embedding+second_node_embedding), 0)
        x =  F.relu(self.first_layer(third_tensor))
        x = F.softmax(self.second_layer(x))
        print ("xxx", x)
        return x
        # print ("x_shape", x.shape)
        #