import torch.nn as nn
import torch.nn.functional as F


class MLP1A(nn.Module):
    """
    Shared Multilayer Perceptron composed of a succession of fully connected layers,
    batch-norms and ReLUs
    INPUT (N x L x C)
    """

    def __init__(self):
        super(MLP1A, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.fc1(x)

        x = x.transpose(2, 1)
        x = self.bn1(x)     # BN1d takes [batch_size x channels x seq_len] 
        x = x.transpose(2, 1)
        x = self.dropout(x) 
        x = F.relu(x)
        return x
    
    # this part is used for layerwise relevance propagation (LRP)
    # dropout layer is left out
    def forward_lrp(self, x):
        x = self.fc1(x)

        x = x.transpose(2, 1)
        x = self.bn1(x)     # BN1d takes [batch_size x channels x seq_len] 
        x = x.transpose(2, 1)
        x = F.relu(x)
        return x


class MLP1B(nn.Module):
    """
    Shared Multilayer Perceptron composed of a succession of fully connected layers,
    batch-norms and ReLUs
    INPUT (N x L x C)
    """

    def __init__(self):
        super(MLP1B, self).__init__()

        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = self.fc2(x)
        x = x.transpose(2, 1)
        x = self.bn2(x)   # BN1d takes [batch_size x channels x seq_len]
        x = x.transpose(2, 1)
        x = F.relu(x)
        return x
    
class MLP2A(nn.Module):
    """
    Multilayer Perceptron number three
    INPUT (N x L)
    Lazylinear is used as the number of infeatures changes with different sequence lengths
    """

    def __init__(self):
        super(MLP2A, self).__init__()
        self.fc1 = nn.LazyLinear(64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)

        return x

    def forward_lrp(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        return x

        
class MLP2B(nn.Module):
    """
    Multilayer Perceptron number three
    INPUT (N x L)
    """

    def __init__(self):
        super(MLP2B, self).__init__()

        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
    



class MLP3A(nn.Module):
    """
    Decoder Multilayer Perceptron.
    INPUT (N x L)
    """

    def __init__(self):
        super(MLP3A, self).__init__()
  
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(num_features=32)


    def forward(self, x):
 
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x
    


class MLP3B(nn.Module):
    """
    Decoder Multilayer Perceptron.
    INPUT (N x L)
    """

    def __init__(self):
        super(MLP3B, self).__init__()

        self.fc3 = nn.Linear(32, 44)


    def forward(self, x):


        x = self.fc3(x)
        return x          
    