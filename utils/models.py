import torch.nn as nn
import torch.nn.functional as F


class linear_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.linear(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class deep_linear_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(deep_linear_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class mlp_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=1):
        super(mlp_model, self).__init__()
        self.ls = []
        previous_dim = input_dim
        for i in range(hidden_layers+1):
            if i < hidden_layers:
                next_dim = hidden_dim
            else:
                next_dim = output_dim

            layer = nn.Linear(previous_dim, next_dim)
            self.dummy = layer
            self.ls.append(layer)
            self.add_module("hidden_layer"+str(i), layer)
            if i < hidden_layers:
                act = nn.ReLU()
                self.add_module("act"+str(i), act)
                self.ls.append(act)
            previous_dim = next_dim
        #self.ls.append(nn.Linear(hidden_dim, output_dim))
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        for layer in self.ls:
            out = layer(out)
        # out = self.fc1(out)
        # out = self.relu1(out)
        # out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet(nn.Module):
    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, (2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
