from Graphs import *
from modelv2 import DCNNv2
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.utils.data
import torch.nn.functional as F
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
import argparse
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--node_representation_size', type=int, default=64)
parser.add_argument('--negative_to_positive_link_ratio', type=float, default=2.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--debug_gradient', type=str2bool, default=False)

args = parser.parse_args()
args_params = vars(args)
node_representation_size = args_params['node_representation_size']
negative_to_positive_link_ratio = args_params['negative_to_positive_link_ratio']
epochs = args_params['epochs']
learning_rate = args_params['learning_rate']
batch_size = args_params['batch_size']
debug_gradient = args_params['debug_gradient']

torch.set_printoptions(threshold=5000)
Graphs.initialize(node_representation_size=node_representation_size, 
                negative_to_positive_link_ratio=negative_to_positive_link_ratio)

train_X, valid_X, test_X, train_y, valid_y, test_y = Graphs.get_train_valid_examples()
train_datasets = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size)   
writer = SummaryWriter()


model = DCNNv2()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print ("Evaluating before training")
y_pred = model(test_X)
loss = F.binary_cross_entropy(y_pred, test_y)
print ("loss on test before training")
print (loss)
for epoch_id in range(epochs):

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)

        loss = F.binary_cross_entropy(y_pred, labels)
        print ("epoch ", epoch_id, " loss", i, loss.item())
        
        optimizer.zero_grad()

        loss.backward()
    
        optimizer.step()
        
        for name, param in model.named_parameters():
            if debug_gradient:
                if param.requires_grad:
                    print (name, "\n", param.data, "\n", "grad", param.grad)
        
    writer.add_pr_curve("pr_curve, epoch_id:" + str(epoch_id), test_y, model(test_X))

    writer.add_scalars('loss', {'training': F.binary_cross_entropy(model(train_X), train_y),
                                'validation': F.binary_cross_entropy(model(test_X), test_y)}, epoch_id)

print ("Evaluating after training")
y_pred = model(test_X)
loss = F.binary_cross_entropy(y_pred, test_y)
print ("loss on test after training")
print (loss)
writer.close()