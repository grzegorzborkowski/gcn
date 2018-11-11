import models
from layers import *
import torch

#model = models.DCNN()

learning_rate = 0.1
graph = Graph()
v1 = Node(torch.rand(1,3))
v2 = Node(torch.rand(1,3))
v3 = Node(torch.rand(1,3))
v4 = Node(torch.rand(1,3))
v5 = Node(torch.rand(1,3))

v1.add_neighbour(v2)
v2.add_neighbour(v3)
v1.add_neighbour(v3)
v3.add_neighbour(v4)
v1.add_neighbour(v4)
v4.add_neighbour(v5)
graph.add_node(v1)
graph.add_node(v2)
graph.add_node(v3)
graph.add_node(v4)
graph.add_node(v5)


model = models.DCNN(graph)
#print (model.igcn1.internal_graph)
for t in range(5):
    print(model.igcn1.internal_graph)
    model()
    #loss = loss_fn(y_pred, train_y)

    if t % 25 == 0:
        pass
        # print (t, loss.item())

    model.zero_grad()
    #loss.backward()

    #with torch.no_grad():
        #for param in model.parameters():
            #param.data -= learning_rate * param.grad
#print (model.igcn1.internal_graph)
#predict_out = model(test_X)
#_, predict_y = torch.max(predict_out, 1)
#accuracy = accuracy_score(test_y, predict_y)
#print(accuracy)
