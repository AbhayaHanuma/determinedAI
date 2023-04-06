#chnaging test
from determined.experimental import client
from determined.keras import load_model_from_checkpoint_path
# from tensorflow import keras


checkpoint = client.get_experiment(131).top_checkpoint()
path = checkpoint.download()
print(path)
model = load_model_from_checkpoint_path(path)

print(model.summary())
# predictions = model(samples)