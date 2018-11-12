import models
from layers import *
import torch


learning_rate = 0.1
graph1 = Graph()
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
graph1.add_node(v1)
graph1.add_node(v2)
graph1.add_node(v3)
graph1.add_node(v4)
graph1.add_node(v5)

graph2 = Graph()
v6 = Node(torch.rand(1,3))
v7 = Node(torch.rand(1,3))
v8 = Node(torch.rand(1,3))
v9 = Node(torch.rand(1,3))
v10 = Node(torch.rand(1,3))
v11 = Node(torch.rand(1,3))
v6.add_neighbour(v7)
v6.add_neighbour(v9)
v7.add_neighbour(v8)
v7.add_neighbour(v9)
v9.add_neighbour(v10)
v10.add_neighbour(v11)
graph2.add_node(v6)
graph2.add_node(v7)
graph2.add_node(v8)
graph2.add_node(v9)
graph2.add_node(v10)
graph2.add_node(v11)

external_graph = Graph()
v1 = Node(torch.rand(1,3))
v2 = Node(torch.rand(1,3))
external_graph.add_node(v1)
external_graph.add_node(v2)
v1.add_neighbour(v2)

model = models.DCNN(graph1, graph2, external_graph)
#print (model.igcn1.internal_graph)
for t in range(5):
    print (model.exgcn1.node_in_external_graph)
    #print(model.igcn1.internal_graph)
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
