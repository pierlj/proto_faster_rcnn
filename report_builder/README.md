## Report builder tool for few-shot Faster R-CNN detector

Computes: 
* Embedding TSNE representation for both RPN and R-CNN
* Per class proposals colored by score in the RPN
* Evaluation metrics for multiple shots

### Options
`--path`: Path to the model to be evaluated
`--store-in-zoo`: Move model in zoo
`--new-name:` name for network
`--copy-training`: Copy training logs next to network.
`--no-details`: Write report without model description
`--no-eval`: Write report without model evaluation