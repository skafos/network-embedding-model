import pickle

def save_embeddings(path, model):
  try:
    embeddings = model.node_embeddings.weight.data.numpy()
    with open(path, 'wb') as f:
      f.write(pickle.dumps(embeddings))
    return True
  except Exception as e:
    print(f'ERROR: {e}')
    return False


def load_embeddings(path):
  try:
    with open(path, 'rb') as f:
      return pickle.loads(f.read())
  except Exception as e:
    print(f'ERROR: {e}')
    return False
