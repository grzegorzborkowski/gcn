from Graphs import *
from modelv2 import DCNNv2
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.utils.data
import torch.nn.functional as F
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
import argparse
from utils import *
import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--node_representation_size', type=int, default=64)
parser.add_argument('--negative_to_positive_link_ratio', type=float, default=2.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--debug_gradient', type=str2bool, default=False)
parser.add_argument('--dataset', type=datasetchoice, default=Dataset.TOY)

args = parser.parse_args()
args_params = vars(args)
node_representation_size = args_params['node_representation_size']
negative_to_positive_link_ratio = args_params['negative_to_positive_link_ratio']
epochs = args_params['epochs']
learning_rate = args_params['learning_rate']
batch_size = args_params['batch_size']
debug_gradient = args_params['debug_gradient']
dataset_path = args_params['dataset']

torch.set_printoptions(threshold=5000)
Graphs.initialize(node_representation_size=node_representation_size,
                negative_to_positive_link_ratio=negative_to_positive_link_ratio,
                dataset_path=dataset_path)

train_X, valid_X, test_X, train_y, valid_y, test_y = Graphs.get_train_valid_examples()
train_datasets = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size)
writer = SummaryWriter()


model = DCNNv2()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print ("Evaluating before training")
y_pred = model(test_X)
y_pred_numpy = y_pred.detach().numpy()
test_y_numpy = test_y.detach().numpy()
loss = F.binary_cross_entropy(y_pred, test_y)
writer.add_scalars('precision/recall/f1', {
                                'precision': precision_score(test_y_numpy, y_pred_numpy > 0.5, average='samples'),
                                'recall': recall_score(test_y_numpy, y_pred_numpy > 0.5, average='samples'),
                                'f1_score': f1_score(test_y_numpy, y_pred_numpy > 0.5, average='samples')}, 0 )
print ("loss on test before training")
print (loss)
for epoch_id in tqdm.tqdm(range(epochs)):

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)

        loss = F.binary_cross_entropy(y_pred, labels)
        if i%50 == 0:
            print ("epoch ", epoch_id, " loss", i, loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        for name, param in model.named_parameters():
            if debug_gradient:
                if param.requires_grad:
                    print (name, "\n", param.data, "\n", "grad", param.grad)

    writer.add_pr_curve("pr_curve, epoch_id:" + str(epoch_id), valid_y, model(valid_X))

    writer.add_scalars('loss', {'training': F.binary_cross_entropy(model(train_X), train_y),
                               'validation': F.binary_cross_entropy(model(valid_X), valid_y)}, epoch_id)

print ("Evaluating after training")
y_pred = model(test_X)
loss = F.binary_cross_entropy(y_pred, test_y)

y_pred_numpy = y_pred.detach().numpy()
test_y_numpy = test_y.detach().numpy()

writer.add_scalars('precision/recall/f1', {
                                'precision': precision_score(test_y_numpy, y_pred_numpy > 0.5, average='samples'),
                                'recall': recall_score(test_y_numpy, y_pred_numpy > 0.5, average='samples'),
                                'f1_score': f1_score(test_y_numpy, y_pred_numpy > 0.5, average='samples')}, 1)
print ("loss on test after training")
print (loss)


# TODO: Make this a separate function, and a parser option for plotting
all_samples = []
for sample in train_X:
    first, second = sample[0].numpy(), sample[1].numpy()
    first_embedding = model.get_node_embedding(first.item()).detach().numpy()
    second_embedding = model.get_node_embedding(second.item()).detach().numpy()
    all_samples.append(first_embedding.T.flatten())
    all_samples.append(second_embedding.T.flatten())

X_embedded = TSNE(n_components=2).fit_transform(all_samples[:1000])
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.show()

####################################################################
model.internal_graph_encoder.visualize_internal_nodes_embedding()

writer.close()