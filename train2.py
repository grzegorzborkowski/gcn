from Graphs import *
from modelv2 import DCNNv2
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.utils.data
import torch.nn.functional as F
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--node_representation_size', type=int, default=64)
parser.add_argument('--negative_to_positive_link_ratio', type=float, default=2.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=8)

args = parser.parse_args()
args_params = vars(args)
node_representation_size = args_params['node_representation_size']
negative_to_positive_link_ratio = args_params['negative_to_positive_link_ratio']
epochs = args_params['epochs']
learning_rate = args_params['learning_rate']
batch_size = args_params['batch_size']

torch.set_printoptions(threshold=5000)
Graphs.initialize(node_representation_size=node_representation_size, 
                negative_to_positive_link_ratio=negative_to_positive_link_ratio)
#Graphs.external_graph.print_graph() #print_external_graph()
# Graphs.get_internal_graph(0).print_graph()
train_X, test_X, train_y, test_y = Graphs.get_train_valid_examples()
train_datasets = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size)
writer = SummaryWriter()


model = DCNNv2()
#loss_fn = F.binary_cross_entropy(size_average=False) # zmienic
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch_id in range(epochs):

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # print (inputs)
        # print (type(inputs))
        y_pred = model(inputs)

    
        loss = F.binary_cross_entropy(y_pred, labels)
        # print (y_pred)
        # print (labels)
        print ("epoch ", epoch_id, " loss", i, loss.item())
        # print (labels)
        
        optimizer.zero_grad()
        # print (optimizer)
        loss.backward()#retain_graph=True)
    
        optimizer.step()
        
    
    
        # print ("predicted_maxed", y_pred)


        for name, param in model.named_parameters():
            pass
            # if param.requires_grad:
                # print (name, "\n", param.data, "\n", "grad", param.grad)
        
        #accuracy = accuracy_score(test_y, predict_y)
   
    writer.add_pr_curve
    print ("evaluating pr score")
    writer.add_pr_curve("pr_curve, epoch_id:" + str(epoch_id), test_y, model(test_X))

    writer.add_scalars('loss', {'training': F.binary_cross_entropy(model(train_X), train_y),
                                'validation': F.binary_cross_entropy(model(test_X), test_y)}, epoch_id)
writer.close()