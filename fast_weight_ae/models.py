import torch
import torch_geometric
from torch_scatter import scatter_mean, scatter_add, scatter_softmax


def get_neighbors(x, edge):
    neighbors = []
    for i in range(x.shape[0]):
        neigh = edge[0, edge[1,:] == i].unique()
        neighbors.append(neigh)
    return neighbors

def get_source_sink(x, edge):
    sources, sinks = [], []
    for i in range(x.shape[0]):
        b_idx = edge[1,:] == i
        source_idx, sink_idx = edge[:, b_idx].unbind(0)
        sources.append(source_idx)
        sinks.append(sink_idx)

    return torch.cat(sources), torch.cat(sinks)


class GNNEncoder(torch.nn.Module):
    """Encode a given neighborhood into a fixed-size latent state
    z and num_neighbor keys. Z can be queried using the keys to reconstruct
    the neighborhood"""
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        latent_size, 
        gnn
    ):
        super().__init__()
        self.act = torch.nn.LeakyReLU(inplace=True)
        self.gnn = gnn
        self.mlp = torch.nn.Sequential(
            self.act,
            torch.nn.Linear(hidden_size, hidden_size),
            self.act,
            torch.nn.Linear(hidden_size, latent_size)
        )
        self.keys = torch.nn.Linear(input_size, latent_size)
        self.weight = torch.nn.Linear(input_size, 1)

    def forward(self, x, edge, **gnn_kwargs):
        z = self.gnn(x, edge, **gnn_kwargs)
        z = self.mlp(z)
        keys = self.keys(x)
        weights = self.weight(x)
        return z, keys, weights


class DotDecoder(torch.nn.Module):
    """Given a latent embedding z and and a set of keys,
    reconstruct the input by querying z with the keys."""
    def __init__(self, input_size, hidden_size, latent_size, use_key_value=False):
        super().__init__()
        self.use_key_value = use_key_value
        if use_key_value:
            self.map = torch.nn.Linear(latent_size, input_size),
        else:
            self.map = torch.nn.Linear(2 * latent_size, input_size)

    def forward(self, z, keys, source_sink, weights=None, use_key_value=False):
        sources, sinks = source_sink
        if self.use_key_value:
            dot = keys[sources] * z[sinks]
            mapped = self.map(dot)
        else:
            mapped = self.map(torch.cat([keys[sources], z[sinks]], dim=-1))
        # Multiply outputs by sigmoidal weights
        # allowing us to assign lower weights
        # where we are uncertain
        # TODO: should we use scatter sum instead?
        if weights is not None:
            importance = weights[sources].sigmoid().reshape(-1)
            mapped = mapped * importance.reshape(-1, 1)
            soft_degree = scatter_add(importance, sources)

        # Reduce by taking the mean over common neighbors
        # e.g. if node 1 is a neighbor of root nodes 3 and  4,
        # vertex 1 will take the mean of 3 and 4
        # 
        # TODO: means are in different frames,
        # this will not work with relative poses
        if weights is not None:
            reduced = scatter_add(mapped, sources, dim=0) / soft_degree.reshape(-1, 1)
        else:
            reduced = scatter_mean(mapped, sources, dim=0)

        return reduced

    def loss(self, x, y_hat, source_sink):
        #sources, sinks = source_sink
        #error = x[sources] - y_hat
        error = x - y_hat
        mse = (error ** 2).mean()
        return mse



class FastWeightDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.key = torch.nn.Linear(input_size, latent_size)
        self.map = torch.nn.Linear(latent_size, input_size)


    def forward(self, z, x, neighbors):
        out_keys, out_values = [], []
        keys = self.key(x)
        
        for i in range(x.shape[0]):
            neigh = neighbors[i]
            key = keys[neigh]
            n, d = key.shape
            fast_weight = z[i].reshape(1, d).repeat(d, 1)
            value = self.map(key @ fast_weight)

            out_keys.append(key)
            out_values.append(value)

        return out_keys, out_values

    def loss(self, x, values, neighbors):
        errors = []
        for i in range(x.shape[0]):
            neigh = neighbors[i]
            error = x[neigh] - values[i]
            errors.append(error)

        loss = (torch.stack(errors) ** 2).mean()
        return loss


