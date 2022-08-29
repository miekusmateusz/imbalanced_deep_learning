from torch import nn


class EmbeddingNet(nn.Module):
    def __init__(self, model_config):
        super(EmbeddingNet, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(model_config['layers_config'][0], model_config['layers_config'][1]),
            nn.PReLU(),
            nn.Linear(model_config['layers_config'][1], model_config['layers_config'][2]),
            nn.PReLU(),
            nn.Linear(model_config['layers_config'][2], model_config['layers_config'][3])
            )

    def forward(self, x):
        output = self.embedding_net(x)
        return output

    def embed(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def embed(self, x):
        return self.embedding_net(x)
