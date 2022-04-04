import torch
import torch_geometric
from models import GNNEncoder, FastWeightDecoder, DotDecoder, get_neighbors, get_source_sink

input_size = 16
duplication_ratio = 2
hidden_size = 16
latent_size = 8
batch_size = 256

gnn = torch_geometric.nn.GraphConv(input_size, hidden_size)
encoder = GNNEncoder(input_size, hidden_size, latent_size, gnn)
decoder = DotDecoder(input_size, hidden_size, latent_size)
opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
for epoch in range(int(1e5)):
    x = torch.cat([
        torch.randn(batch_size, input_size // duplication_ratio)
    ] * duplication_ratio, dim=-1)
    #edge = torch_geometric.nn.radius_graph(x, r=2)
    edge = torch_geometric.nn.knn_graph(x, k=3)
    # Self loops required for decode scatter
    edge, _ = torch_geometric.utils.add_remaining_self_loops(edge)
    z, keys, weights = encoder(x, edge)
    source_sink = get_source_sink(x, edge)
    y = decoder(z, keys, weights, source_sink)
    loss = decoder.loss(x, y, source_sink)
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    




