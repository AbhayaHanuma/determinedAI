import numpy as np
import pandas as pd

from determined.experimental import client
from determined.keras import load_model_from_checkpoint_path
from fetch import load_training_data

checkpoint = client.get_experiment(18).top_checkpoint()
path = checkpoint.download()
# print(f'checkpoint path: {path}')
model = load_model_from_checkpoint_path(path)

model.summary()
x_train, y_train, scaler, _, last_win_data= load_training_data(window=12,val=0)
# model.fit(x_train,y_train,epochs=1000)
predict_norm = model.predict(last_win_data)

predict_denorm = scaler.inverse_transform(np.hstack((last_win_data[-1,-1:,:-1],predict_norm)))
print(predict_denorm[-1][-1])