### Dual Convolutional Neural Network for Graph of Graphs Link Prediction training script


#### Usage

```
usage: train2.py [-h] [--node_representation_size NODE_REPRESENTATION_SIZE]
                 [--negative_to_positive_link_ratio NEGATIVE_TO_POSITIVE_LINK_RATIO]
                 [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                 [--batch_size BATCH_SIZE] [--debug_gradient DEBUG_GRADIENT]
                 [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --node_representation_size NODE_REPRESENTATION_SIZE
  --negative_to_positive_link_ratio NEGATIVE_TO_POSITIVE_LINK_RATIO
  --epochs EPOCHS
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --debug_gradient DEBUG_GRADIENT
  --dataset DATASET
```

#### Parameters

optional arguments: <br>
  -h, --help            show this help message and exit <br>
  --node_representation_size NODE_REPRESENTATION_SIZE. Size of the embedding for internal node <br>
  --negative_to_positive_link_ratio NEGATIVE_TO_POSITIVE_LINK_RATIO How many negative (non existing) links add to training. Default 2<br>
  --epochs EPOCHS  (number of training epochs). Default 20 <br>
  --learning_rate LEARNING_RATE. Default 0.1 <br>
  --batch_size BATCH_SIZE Default 8<br>
  --debug_gradient DEBUG_GRADIENT Default False. Prints information about gradient after each training batch<br>
  --dataset DATASET Default toy. Which dataset to choose to training Options: toy/dbpl/drugs. <br>


Model described here: https://arxiv.org/pdf/1810.02080.pdf