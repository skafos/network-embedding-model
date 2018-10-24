import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def create_batch(edge_samples, edge_weight_dict, K, node_sampler):
  """Given a set of sampled edges, generate a training batch for the embedding model.
       :param: edge_samples- list or array-like of tuples. Each tuple contains paired integers refering to (src_node, dst_node).
       :param: edge_weight_dict- dictionary. Keys refer to node integer pairs, Values refer to their relative weight.
       :param: K- integer. Number of negative samples to draw for each oberserved edge.
       :param: node_sampler- object of class <Sampler()>. Used to draw negative node samples.

       Returns:
       - v_i = Torch Tensor of source node ids.
       - v_j = Torch Tensor of destination node ids.
       - labels = Torch Tensor of ground-truth labels (-1, 1).
    """
  v_i = []
  v_j = []
  labels = []
  # For each sampled edge, negatively sample K nodes, and add v_i, v_j, and label to the batch
  for edge in edge_samples:
    src_node, dst_node = edge[0], edge[1]
    v_i.append(src_node)
    v_j.append(dst_node)
    labels.append(1)
    # Generate K negative samples for each observed edge
    for _ in range(K):
      while True:
        neg_sample_node = node_sampler.sample_n(1)[0]
        # Check to see if this proposed negative edge exists or not
        if edge_weight_dict.get((src_node, neg_sample_node)) is None:
          break
      # If we found a good negative edge, add to the batch!
      v_i.append(src_node)
      v_j.append(neg_sample_node)
      labels.append(-1)
  return torch.LongTensor([v_i]), torch.LongTensor([v_j]), torch.FloatTensor([labels])


class Line(nn.Module):
  def __init__(self, size, embedding_dim=128, order=1):
    super(Line, self).__init__()

    assert order in [1, 2], print("Order should either be int(1) or int(2)")
    # Number of nodes we are trying to embed
    self.size = size
    # Output dimension of each embedding vector v
    self.embedding_dim = embedding_dim
    # First-order OR Second-order (we will focus on first-order for the class)
    self.order = order
    print(f'Initing a {order}-order, {embedding_dim} dimensional embedding model for {size} nodes', flush=True)
    self.node_embeddings = nn.Embedding(self.size, self.embedding_dim)

    # Initialize node embeddings between -1 and 1, pulled from a uniform random dist
    self.node_embeddings.weight.data = F.normalize(self.node_embeddings.weight.data.uniform_(-1., 1), p=2, dim=1)

    # If trying second-order, initialize context embeddings between -1 and 1, pulled from a uniform random dist
    if order == 2:
      self.contextnode_embeddings = nn.Embedding(self.size, self.embedding_dim)
      self.contextnode_embeddings.weight.data = F.normalize(self.contextnode_embeddings.weight.data.uniform_(-1., 1), p=2, dim=1)

  def forward(self, v_i, v_j, labels, batch_size):
    # Find tensors v_i and v_j from the embedding matrix
    v_i = self.node_embeddings(v_i)

    # If trying second-order, v_j comes from the context embeddings
    if self.order == 2:
      v_j = self.contextnode_embeddings(v_j)
    else:
      v_j = self.node_embeddings(v_j)

    # Step 1: Calculate the inner product of embedding vector pairs (i, j), (i, n_1), ... (i, n_k) for K negative node samples
    ## This will return a single vector of scalars with length equal to (1+K)*batch_size
    inner_prod = torch.sum(torch.mul(v_i, v_j), dim=2)

    # Step 2: Loss is calculated by taking the log sigmoid of the inner product from step 1
    ## But first, we multiply by the labels representing ground truth (-1, 1)
    loss = F.logsigmoid(torch.mul(labels, inner_prod))

    # Return the average over all batches, and ensure that loss is positive
    ### HINT: motivation to keep the batch size smaller.. SGD works the way it's supposed to this way
    return -torch.sum(loss) / batch_size


class Sampler(object):
  """Basic sampler class given some items and discrete probabilities"""

  def __init__(self, items, probabilities):
    self.dist = stats.rv_discrete(values=(items, probabilities))

  def sample_n(self, n):
    if isinstance(n, int):
      return list(self._sample(n))
    else:
      print('You must pass an integer value!', flush=True)

  def _sample(self, n):
    for _ in range(n):
      yield self.dist.rvs()
