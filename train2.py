from Graphs import *
from modelv2 import DCNNv2
import torch.nn as nn

Graphs.initialize()
#Graphs.external_graph.print_graph() #print_external_graph()
# Graphs.get_internal_graph(0).print_graph()
train_X, test_X, train_y, test_y = Graphs.get_train_valid_examples()
model = DCNNv2()
loss_fn = nn.CrossEntropyLoss() # zmienic

learning_rate = 0.1

for t in range(250):
    y_pred = model(train_X)
    loss = loss_fn(y_pred, train_y)

    if t % 25 == 0:
        print (t, loss.item())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad

    predict_out = model(test_X)
    _, predict_y = torch.max(predict_out, 1)
    accuracy = accuracy_score(test_y, predict_y)
    print(accuracy)
