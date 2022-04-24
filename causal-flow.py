import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import distributions
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

n_samples = 512
selected_dataset = 'causal'  # in {'causal', 'moons'}
epochs = 5001
num_flows = 1
learning_rate = 1e-3
hidden_dim = 8
test_size = 0.3
batch_size = n_samples // 8

if selected_dataset == 'causal':
    causal_func = 'non-linear'  # in {'linear', 'non-linear'}
    noise = 'laplace'  # in  {'gaussian', laplace, 'cauchy', 'student'}

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dim = 4

# mask_x_y = [[0, 1],
#             [0, 0]]
#
# mask_y_x = [[0, 0],
#             [1, 0]]

# mask_x_y = [[0, 1, 1],
#             [0, 0, 0],
#             [0, 0, 0]]
#
# mask_y_x = [[0, 0, 0],
#             [1, 0, 0],
#             [1, 0, 0]]

mask_1 = [[0, 1, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 1],
          [0, 0, 0, 0]]

mask_2 = [[0, 0, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 0, 0],
          [0, 1, 1, 0]]

mask_3 = [[0, 0, 1, 0],
          [0, 0, 1, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

mask_4 = [[0, 0, 1, 1],
          [1, 0, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]]

mask_5 = [[0, 1, 1, 1],
          [0, 0, 1, 1],
          [0, 0, 0, 1],
          [0, 0, 0, 0]]

masks_dict = {'true-dag': [mask_1 for _ in range(num_flows)],
              # 'reverse-dag': [mask_2 for _ in range(num_flows)],
              'false-dag': [mask_3 for _ in range(num_flows)],
              'complete-dag': [mask_5 for _ in range(num_flows)]}


# masks_dict = {'1': [mask_1 for _ in range(num_flows)],
#               '2': [mask_1 for _ in range(num_flows)],
#               '3': [mask_1 for _ in range(num_flows)],
#               '4': [mask_1 for _ in range(num_flows)]}


class RealNVP(nn.Module):
    def __init__(self, masks, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.masks = nn.Parameter(masks, requires_grad=False)
        self.dim = masks[0].shape[0]
        self.masks_roots = torch.from_numpy(
            np.array([self._find_roots(mask.T) for mask in masks]).astype(np.float32)).to(device)

        nets = lambda: nn.Sequential(nn.Linear(dim, hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_dim, 1),
                                     nn.Tanh())

        nett = lambda: nn.Sequential(nn.Linear(dim, hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_dim, 1))

        self.t = torch.nn.ModuleList([])
        self.s = torch.nn.ModuleList([])

        for i in range(self.dim):
            dependent = self.A[i, :]

            for i, param in enumerate(m.parameters()):
                if i == 0:
                    param.grad[:, 1:] = torch.zeros_like(param.grad[:, 1:])

            self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
            self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    # def g(self, z):
    #     x = z
    #     for i in range(len(self.t)):
    #         x_ = x * self.masks[i]
    #         s = self.s[i](x_) * (1 - self.masks[i])
    #         t = self.t[i](x_) * (1 - self.masks[i])
    #         x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)
    #     return x

    @staticmethod
    def _find_roots(mask):
        mask_numpy = mask.detach().cpu().numpy()
        return ~mask_numpy.any(axis=1) * 1

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = torch.matmul(self.masks[i].T, z.T).T

            s = self.s[i](z_) * (1 - self.masks_roots[i])

            # t = self.t[i](z_) * (1 - self.masks[i])
            t = self.t[i](z_) * (1 - self.masks_roots[i])

            # z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            z = (1 - self.masks_roots[i]) * (z - t) * torch.exp(-s) + self.masks_roots[i] * z
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


def gen_synth_causal_dat(n_samples=100, noise=0.05, causalFunc=causal_func, noise_dist=noise):
    causalFuncDict = {'linear': lambda x, n: 10 * x + n,
                      'non-linear': lambda x, n: x + (.5) * x * x * x + (n),
                      # 'nueralnet_l1': lambda x, n: sigmoid(sigmoid(np.random.normal(loc=1) * x) + n),
                      # 'mnm': lambda x, n: sigmoid(np.random.normal(loc=1) * x) + .5 * x ** 2
                      #                     + sigmoid(np.random.normal(loc=1) * x) * n
                      }

    # scale divided by np.sqrt(2) to ensure std of 1
    if noise_dist == 'laplace':
        N = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_samples, dim))
    elif noise_dist == 'gaussian':
        N = np.random.normal(loc=0, scale=1., size=(n_samples, dim))
    elif noise_dist == 'cauchy':
        N = np.random.standard_cauchy(size=(n_samples, dim))
    elif noise_dist == 'student':
        N = np.random.standard_t(df=5, size=(n_samples, dim))
    else:
        raise ValueError(noise_dist)

    X = np.zeros((n_samples, dim))
    X[:, 0] = N[:, 0]
    X[:, 1] = causalFuncDict[causalFunc](X[:, 0], N[:, 1])
    X[:, 2] = causalFuncDict[causalFunc](X[:, 0], N[:, 2])
    X[:, 3] = causalFuncDict[causalFunc](X[:, 1] + X[:, 2], N[:, 3])

    return X, None


datasets = {'moons': make_moons,
            'causal': gen_synth_causal_dat}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_flow(train_loader, validation_loader, mask):
    masks = torch.from_numpy(np.array(mask).astype(np.float32)).to(device)
    prior = distributions.MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))
    flow = RealNVP(masks, prior).to(device)
    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=learning_rate)
    # optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=learning_rate,
    #                              weight_decay=0.00, betas=(0.9, 0.999), amsgrad=False)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    train_loss = []
    validation_loss = []

    for t in range(epochs):
        train_batches_loss = []
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            loss = -flow.log_prob(x_batch).mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_batches_loss.append(loss.item())

        train_loss.append(np.mean(train_batches_loss))

        with torch.no_grad():
            validation_batches_loss = []
            for x_val, _ in validation_loader:
                x_val = x_val.to(device)
                loss_val = -flow.log_prob(x_val).mean().item()
                validation_batches_loss.append(loss_val)

            validation_loss.append(np.mean(validation_batches_loss))

        if t % 1000 == 0:
            print(f'epoch: {t}, train loss = {round(train_loss[-1], 3)},'
                  f' validation loss = {round(validation_loss[-1], 3)}')

        # scheduler.step(loss_val)
    print('-' * 80)
    return flow, train_loss, validation_loss


train_losses = {}
validation_losses = {}
flows = {}
noisy_dataset = datasets[selected_dataset](n_samples, noise=0.05)[0].astype(np.float32)
train, validation = train_test_split(noisy_dataset, test_size=test_size, random_state=seed)
train = torch.from_numpy(train).to(device)
validation = torch.from_numpy(validation).to(device)
train_dataset = TensorDataset(train, train)
validation_dataset = TensorDataset(validation, validation)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

for mask_name, mask in masks_dict.items():
    flow, train_loss, validation_loss = train_flow(train_loader, validation_loader, mask)
    train_losses[mask_name] = train_loss
    validation_losses[mask_name] = validation_loss
    flows[mask_name] = flow


def plot_losses(train_losses, validation_losses):
    colors = ['black', 'blue', 'red', 'teal', 'yellow']
    plt.figure()
    for i, (name, loss) in enumerate(train_losses.items()):
        # plt.plot(np.log(loss), label=name, c=colors[i])
        plt.plot(loss[1:], label=name, c=colors[i])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'Training loss - different masks comparison - {selected_dataset} dataset \n n_samples = {n_samples}')
    plt.legend()

    plt.figure()
    for i, (name, loss) in enumerate(validation_losses.items()):
        # plt.plot(np.log(loss), label=name, c=colors[i])
        plt.plot(loss[1:], label=name, c=colors[i])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'Validation loss - different masks comparison - {selected_dataset} dataset \n n_samples = {n_samples}')
    plt.legend()

    plt.show()


plot_losses(train_losses, validation_losses)

# datasets = {'moons': make_moons(n_samples=n_samples, noise=0.05)[0].astype(np.float32),
#             'causal': gen_synth_causal_dat(n_samples=n_samples)[0].astype(np.float32)}

# def plot_data(dataset, flow, flow_name):
#     z = flow.f(torch.from_numpy(dataset).to(device))[0].cpu().detach().numpy()
#     fig, _ = plt.subplots(2, 2)
#     fig.suptitle(f"{selected_dataset} Dataset - {flow_name} - n_samples = {n_samples}", fontsize=16)
#
#     plt.subplot(221)
#     plt.scatter(z[:, 0], z[:, 1])
#     plt.title(r'$z = f^-1(X)$ ' + 'inverse of data to latent')
#
#     z = np.random.multivariate_normal(np.zeros(2), np.eye(2), n_samples)
#     plt.subplot(222)
#     plt.scatter(z[:, 0], z[:, 1])
#     plt.title(r'$z \sim p(z)$ ' + 'sample from latent')
#
#     plt.subplot(223)
#     plt.scatter(dataset[:, 0], dataset[:, 1], c='r')
#     plt.title(r'$X \sim p(X)$ ' + 'sample from data')
#
#     plt.subplot(224)
#     x = flow.sample(n_samples).cpu().detach().numpy()
#     plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
#     plt.title(r'$X = f(z)$ ' + 'generated from latent')
#
#
# for flow_name, flow in flows.items():
#     plot_data(datasets[selected_dataset], flow, flow_name)

plt.show()
