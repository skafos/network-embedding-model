# Network Embedding Model
This example trains a 128-dimensional network embedding model based on the [**LINE Algorithm**](https://arxiv.org/abs/1503.03578) on [**Skafos**](https://docs.metismachine.io/docs/skafos-components). Data is hosted in a 3rd party Graph DB (Memgraph). The technique follows from the 10/31/18 ODSC West workshop "Intro to Network Embeddings". Access to the data in the graph requires connection credentials that were provided in the workshop.

Attendees of the workshop already will have:
- A Skafos login
- Workshop sample code in a jupyter lab instance and lecture slides
- Client graph cnx credentials

 

## Setup
You will need to install the [**Skafos CLI**](https://docs.metismachine.io/docs/installation) (mac or linux), Git, and have python >= 3.6.

###1. Clone or Fork this Repo

###2. Examine `metis.config.yml.example`
Each Skafos project requires its own unique project token and `metis.config.yml` file.
The `metis.config.yml.example` file provided in this repo is just a sample, but is identical in structure to what you will need. In summary, [**the config file**](https://docs.metismachine.io/docs/installation) is the workhorse that controls how your deployment runs.

In the next step, you wil generate your own.
###3: Initialize the Skafos project
Once in the working directory of this project, type:
```bash
skafos init
```
on the command line. This will generate a fresh `metis.config.yml` file that is tied to your Skafos account and organization.
Open up this config file and edit the **name** and **entrypoint** of the existing job to match the example config provided. The job id and project token are unique to you. Add any other options arguments that you might need.

###4: Set Graph Connection ENV VARS
In the ODSC workshop, you received some graph connection credentials (`GRAPH_HOST`, `GRAPH_USER`, `GRAPH_PASSWORD`). Set those in your deployment environment with the CLI:
```bash
skafos env GRAPH_HOST --set <value>
```
###5: Edit the train.py or common/model.py files as you wish
Provided is a `train.py` file that will train the standard model. Feel free to edit any of the entrypoint arguments. Below is the full set of options:

```python
# Available Entrypoint Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-save", "--save_path", type=str, default=None)       # S3 path to save data to. Make sure you have AWS credentials set as env vars.
parser.add_argument("-order", "--order", type=int, default=1)             # First order or Second order proximity.
parser.add_argument("-neg", "--negsamplesize", type=int, default=5)       # Number of negative samples. Original value from the LINE paper.
parser.add_argument("-dim", "--dimension", type=int, default=128)         # Number of output embedding dimensions. Original value from the LINE paper.
parser.add_argument("-batchsize", "--batchsize", type=int, default=16)    # Size of a training batch.
parser.add_argument("-epochs", "--epochs", type=int, default=5)           # Number of total passes through training data.
parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)  # Initial learning rate. Original value from the LINE paper.
parser.add_argument("-log", "--log", type=int, default=1000)              # Logging interval.
parser.add_argument("-threads", "--threads", type=int, default=1)         # Number of threads for PyTorch to perform computations. Make sure you declare enough CPUs in the metis config file.
args = parser.parse_args()
```

You can run this deployment with all of the defaults.. (but note that it won't save your outputs anywhere!)

###6: Deploy
For the first time only, run the following in your terminal:
```bash
skafos remote info
```
and add the provided remote with the `git remote add ...` command.

After adding the skafos remote, deploy by committing your changes and running `git push skafos <branch-name>`. This will launch a deployment, build dependencies, and orchestrate all infrastructure components automatically.

###7: Monitor
Head over to the [**Dashboard**](https://dashboard.metismachine.io) and checkout your running deployment for this project. A couple live-training metrics, system performance, and logs are available for your review. 

###8: Make some new changes and try again
Data science is all about iteration. After watching your model train.. maybe you aren't satisfied..
Edit different components of the model, training loops, data prep, or Skafos [**User-Defined-Metrics**](https://docs.metismachine.io/docs/skafos-sdk#section-model-monitoring-user-defined-metrics) and [**Logs**](https://docs.metismachine.io/docs/skafos-sdk#section-logging).

Re-deploy your model by committing and pushing changes with git to the skafos remote. (`git push skafos <branch-name>`)

That's it! No need to focus on infrastructure..

## What's Next
Pretty neat? If you have other ways you want to leverage machine learning, checkout [**AddOns**](https://docs.metismachine.io/docs/addons) and other [**SDK Features**](https://docs.metismachine.io/docs/skafos-sdk) provided with Skafos *out-of-the-box*.

Reach out to the [**Skafos Slack Channel**]() with any questions or concerns. We'd also love to get your feedback!
