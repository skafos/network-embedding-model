# Import some python tools
import argparse
import pickle
import s3fs
import numpy as np
from time import time
from skafossdk import *

import torch
import torch.optim as optim
from scipy import stats

from graph.client import Graph
from common.model import Sampler, Line, create_batch

# Parse entrypoint args
parser = argparse.ArgumentParser()
parser.add_argument("-save", "--save_path", type=str, default=None)
parser.add_argument("-order", "--order", type=int, default=1)
parser.add_argument("-neg", "--negsamplesize", type=int, default=5)
parser.add_argument("-dim", "--dimension", type=int, default=128)
parser.add_argument("-batchsize", "--batchsize", type=int, default=16)
parser.add_argument("-epochs", "--epochs", type=int, default=5)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)
parser.add_argument("-log", "--log", type=int, default=1000)
parser.add_argument("-threads", "--threads", type=int, default=1)
args = parser.parse_args()


# Let's connect to the graph!
ska = Skafos()
graph = Graph(skafos=ska)

if not args.save_path:
    ska.log('Heads up, your model will indeed train, but will not save outputs anywhere! Set an s3 path..', labels=['warning'])

# How many edges (relationships) are in the graph?
total_edges = graph.query("MATCH ()-[edge]->() RETURN COUNT(edge)")
total_edges = total_edges[0].get('COUNT(edge)')
print(f'There are {total_edges} edges in the graph', flush=True)

# Calculate the relative "weight" of each node
ska.log('Fetching node weights', labels=['node-weights'])
node_weights = graph.query("""
    MATCH (n)
    WITH sum(degree(n)) as degreeSum
    MATCH (n)
    RETURN n.i as node, toFloat(degree(n))/degreeSum AS weight
""")
node_weights = {n['node']: float(n['weight']) for n in node_weights}

# Calculate the relative "weight" of each edge.. stored as a tuple (node1, node2)
ska.log('Fetching edge weights', labels=['edge-weights'])
edge_weights = graph.query("""
    MATCH ()-[e]->()
    RETURN e.start_i as start_node, e.end_i as end_node, COUNT(e) as weight
""")
edge_weights = {(e['start_node'], e['end_node']): float(e['weight']/total_edges) for e in edge_weights}

# We also need a way to convert a tuple back to an integer key.. because data
edge_map = {}
r = 0
for edge in edge_weights:
    edge_map[r] = edge
    r += 1

# Build discrete probability distributions for both edges and nodes
ska.log('Building discrete probability distribution over nodes and edges!', labels=['prob_dist'])
node_dist = stats.rv_discrete(values=(list(node_weights.keys()), list(node_weights.values())))
edge_dist = stats.rv_discrete(values=(list(edge_map.keys()), list(edge_weights.values())))


# Declare some modeling constants
max_index = len(node_weights) - 1                  # how many nodes to embed
batch_size = int(args.batchsize)                   # size of a single training batch to calculate a loss over
negsamplesize = int(args.negsamplesize)            # number of negative edges to pull for each observed edge
epochs = int(args.epochs)                          # number of passes to make through the training data
num_batches = int(len(edge_weights) / batch_size)  # within an epoch, the number of training steps to make
embed_dim = int(args.dimension)                    # dimension of our output embeddings
order = int(args.order)                            # which proximity to model (from LINE)
learning_rate = float(args.learning_rate)          # initial learning rate that will decay with iteration
log_interval = int(args.log)                       # how often to print out summary information and reset running loss
total_iterations = num_batches*epochs              # total number of training steps
it = 0                                             # init iteration counter to zero
total_loss = 0                                     # init running loss tracker to zero
batch_times = []                                   # bin for collecting batch training times


# Setup up our sampling distributions
nodes = Sampler(
    items=list(node_weights.keys()),
    probabilities = list(node_weights.values())
)
edges = Sampler(
    items = list(edge_map.keys()),
    probabilities = list(edge_weights.values())
)

# Init model and optimizer
line = Line(
    size=max_index + 1,
    embedding_dim=embed_dim,
    order=2
)
optimizer = optim.SGD(
    line.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True
)


ska.log(f'Setting the number of threads to {args.threads}', labels=['embeddings'])
torch.set_num_threads(args.threads)

ska.log(f'Beginning training over {epochs} epochs', labels=['embeddings'])
full_start = time()

for epoch in range(epochs):
    ska.log(f'Epoch {epoch + 1}', labels=['training'])
    for b in range(num_batches):

        batch_start = time()

        # Adjust learning rate after the first couple thousand iterations
        if it < 100000:
          new_lr = learning_rate * (1 - (it / total_iterations))
          for param_group in optimizer.param_groups:
              param_group['lr'] = new_lr

        # Step 1: Draw a sample of edges (batchsize)
        sampled_edges = [edge_map[sample_edge] for sample_edge in edges.sample_n(batch_size)]

        # Step 2: Create a batch of training data including negative samples
        v_i, v_j, labels = create_batch(
            edge_samples=sampled_edges,
            edge_weight_dict=edge_weights,
            K=negsamplesize,
            node_sampler=nodes
        )

        # Step 3: Zero out the accumulated gradients first
        line.zero_grad()

        # Step 4: Pass batch through the LINE algorithm, calculate loss
        loss = line(v_i, v_j, labels, batch_size)

        # Step 5: Calculate gradients and update embedding weights
        loss.backward()
        optimizer.step()

        # Step 6: Update loss tracker
        total_loss += loss.data.item()
        it += 1
        batch_finish = time()
        batch_time = batch_finish-batch_start
        batch_times.append(batch_time)

        if it % log_interval == 0:
            mean_batch_time = float(np.mean(batch_times))
            ska.log(f'Epoch {epoch+1}, Iteration {it}, RunningLoss {total_loss}, Mean Batch Time {mean_batch_time}', labels=['training'])
            # Skafos User Defined Metrics: viewable live on the dashboard (dashboard.metismachine.io) while running
            ska.report('Batch Train Time', y=mean_batch_time, y_label='Avg Batch Completion Time')
            ska.report('Running Model Loss', y=total_loss, y_label='Running Loss')
            batch_times.clear()
            total_loss = 0

ska.log(f'Finished training in {time()-full_start}!', labels=['training'])

# Save model weights to s3 if save path given
if args.save:
    # Fetch embeddings from the current state
    embeddings = line.node_embeddings.weight.data.numpy()
    if 's3://' in args.save:
        ska.log(f'Saving embeddings matrix to {args.save} on S3.', labels=['save'])
        try:
            s3 = s3fs.S3FileSystem(anon=False)
            with s3.open(args.save, 'wb') as f:
                f.write(pickle.dumps(embeddings))
            ska.log(f'Finished writing output to s3.', labels=['save'])
        except Exception as e:
            ska.log(f'Save to s3 failed... do you have AWS Credentials set in your env?', labels=['save'])
else:
    ska.log('No output path given, not saving, sorry..', labels=['save'])

ska.log('DONE', labels=['embeddings'])
