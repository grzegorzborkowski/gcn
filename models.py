import torch.nn as nn
import torch.nn.functional as F
from layers import InternalGraphConvolutionLayer

class DCNN(nn.Module):
    def __init__(self, first_graph): #second_graph, node_representation_size):
        super(DCNN, self).__init__()

        self.igcn1 = InternalGraphConvolutionLayer(first_graph, 3)
        #selc.igcn2 = InternalGraphConvolutionLayer(second_graph, node_representation_size)
        self.layers = nn.ModuleList()
        self.layers.append(self.igcn1)
        #self.layers.append(self.igcn2)

    def forward(self):
        self.igcn1.forward()
        #second = self.igcn2()
        #print (second)
