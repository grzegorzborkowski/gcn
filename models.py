import torch.nn as nn
import torch.nn.functional as F
from layers import *

class DCNN(nn.Module):
    def __init__(self, first_graph, second_graph, external_graph): #second_graph, node_representation_size):
        super(DCNN, self).__init__()

        self.igcn1 = InternalGraphConvolutionLayer(first_graph, 3)
        self.irl1 = InternalRepresentationLayer()

        self.igcn2 = InternalGraphConvolutionLayer(second_graph, 3)
        self.irl2 = InternalRepresentationLayer()

        self.exgcn1 = ExternalGraphConvolutionLayer(external_graph.nodes[0], 3)
        self.exgcn2 = ExternalGraphConvolutionLayer(external_graph.nodes[1], 3)

        self.link_predcition_layer = LinkPredictionLayer(external_graph)
        #self.layers.append(self.igcn1)
        #self.layers.append(self.igcn2)

    def forward(self):
        self.igcn1.forward()
        self.irl1.internal_graph = self.igcn1.internal_graph
        self.exgcn1.node_in_external_graph.representation = self.irl1.forward()
        self.exgcn1.forward()
        return self.link_predcition_layer.forward(0, 1)
        #second = self.igcn2()
        #print (second)
