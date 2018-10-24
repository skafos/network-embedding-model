"""Client connection to a hosted memgraph"""
import os
import numpy as np
from time import sleep
from neo4j.v1 import GraphDatabase
from neo4j.exceptions import TransientError, ClientError


class Graph:

    def __init__(self, skafos):
        self.ska = skafos
        self._authorize()
        self.ska.log("Connecting to memgraph host", labels=['graph'])

        # Initialize and configure the driver.
        #   * provide the correct URL where Memgraph is reachable;
        #   * use an empty user name and password.
        self.driver = GraphDatabase.driver(self.host, auth=(self.user, self.password))
        
    def _authorize(self):
        self.ska.log("Checking environment", labels=['graph'])
        self.host = os.getenv("GRAPH_HOST")
        self.user = os.getenv("GRAPH_USER")
        self.password = os.getenv("GRAPH_PASSWORD")
        
    #def _execute(self, tx, query_string):
    #    for record in tx.run(query_string):
    #        print(record)
            
    def query(self, query_string, label='graph'):
        # sleep 10 seconds to simulate randomized querying from class
        #sleep(np.random.randint(10))
        res = None
        with self.driver.session() as session:
            try:
                res = session.run(query_string).data()
            except TransientError as e:
                self.ska.log(f'TRANSIENT ERROR: {e}', labels=[label])
            except ClientError as e:
                self.ska.log(f'ERROR: {e}', labels=[label])
            except Exception as e:
                self.ska.log(f'ERROR: {e}', labels=[label])
        return res
    
    def sample_edge(self, n=1, edge_type=None):
        try:
            n = int(n)
        except Exception as e:
            self.ska.log(f'ERROR: {e} - you need to pass an int')
        # Draw A Sample    
        if not edge_type:
            return self.query(f"MATCH ()-[edge]->() RETURN edge ORDER BY rand() LIMIT {n}")
        elif edge_type == 'POSTED':
            return self.query(f"MATCH ()-[edge:POSTED]->() RETURN edge ORDER BY rand() LIMIT {n}")
        elif edge_type == 'COMMENTED':
            return self.query(f"MATCH ()-[edge:COMMENTED]->() RETURN edge ORDER BY rand() LIMIT {n}")
        elif edge_type == 'REPLIED':
            return self.query(f"MATCH ()-[edge:REPLIED]->() RETURN edge ORDER BY rand() LIMIT {n}")
        elif edge_type == 'TAGGEDAS':
            return self.query(f"MATCH ()-[edge:TAGGEDAS]->() RETURN edge ORDER BY rand() LIMIT {n}")
        else:
            self.ska.log('ERROR: You passed an incorrect edge type..', labels=['graph'])
