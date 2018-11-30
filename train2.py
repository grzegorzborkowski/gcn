from Graphs import *
from modelv2 import DCNNv2
import torch.nn as nn
from sklearn.metrics import accuracy_score

Graphs.initialize()
#Graphs.external_graph.print_graph() #print_external_graph()
# Graphs.get_internal_graph(0).print_graph()
train_X, test_X, train_y, test_y = Graphs.get_train_valid_examples()
print (train_y)
model = DCNNv2()
loss_fn = nn.BCELoss() # zmienic
print (train_y)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(5):
    y_pred = model(train_X)
    #print (y_pred)
    #print (y_pred.shape)
    loss = loss_fn(y_pred, train_y)

    if t % 25 == 0:
        print (t, loss.item())

    model.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    #with torch.no_grad():
    #    for name, param in model.named_parameters():
    #        print (name, param, "data", param.data)
    ##        print ("param_grad" + str(param.grad))
     #       param.data -= learning_rate * param.grad

    predict_out = model(test_X)
    _, predict_y = torch.max(predict_out, 1)
    print(test)
    print (test_y)
    
    print ("predicted")
    print (predict_y)
    accuracy = accuracy_score(test_y, predict_y)
    print("accuracy" + str(accuracy))
