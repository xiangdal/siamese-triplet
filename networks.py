import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, projection_layers=[256, 256, 2], use_bn=False):
        super(EmbeddingNet, self).__init__()
        modules = [nn.Conv2d(1, 32, 5)]
        if use_bn:
            modules.append(nn.BatchNorm2d(32))
        modules.extend([
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5)])
        if use_bn:
            modules.append(nn.BatchNorm2d(64))
        modules.extend([nn.PReLU(),
            nn.MaxPool2d(2, stride=2)])
        self.convnet = nn.Sequential(*modules)

        proj_layers = []
        input_dim = 64 * 4 * 4
        for i, out_dim in enumerate(projection_layers):
            proj_layers.append(nn.Linear(input_dim, out_dim))
            if use_bn:
                modules.append(nn.BatchNorm1d(out_dim))
            input_dim = out_dim
            if i < (len(projection_layers) - 1):
                proj_layers.append(nn.PReLU())

        self.fc = nn.Sequential(*proj_layers)

        # dimension of the embedding space
        if projection_layers:
            self.embd_dim = projection_layers[-1]
        else:
            self.embd_dim = input_dim

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(embedding_net.embd_dim, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
