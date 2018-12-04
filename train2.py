from Graphs import *
from modelv2 import DCNNv2
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.utils.data
import torch.nn.functional as F
from torch.autograd.variable import Variable

torch.set_printoptions(threshold=5000)
Graphs.initialize()
#Graphs.external_graph.print_graph() #print_external_graph()
# Graphs.get_internal_graph(0).print_graph()
train_X, test_X, train_y, test_y = Graphs.get_train_valid_examples()
train_datasets = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=8)

model = DCNNv2()
#loss_fn = F.binary_cross_entropy(size_average=False) # zmienic
learning_rate = 0.001
epochs = 20
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
        
    
        # print ("predicted_out", predict_out)
    
        #print ("predicted_maxed", predict_y)

        # print ("predicted")
        # print (predict_y)
        # print (predict_y.shape)
        # print ("test_y", test_y)
        # print (test_y.shape)

        for name, param in model.named_parameters():
            pass
            # if param.requires_grad:
                # print (name, "\n", param.data, "\n", "grad", param.grad)
        
        #accuracy = accuracy_score(test_y, predict_y)
        if i % 25 == 0:
            pass
            #predict_out = model(test_X)
            #predict_y = torch.round(predict_out)
            #correct = (predict_y == test_y).float().sum() 
            #print ("accuracy", correct.item()/(len(test_y)*2))
        #print("accuracy", str(accuracy))