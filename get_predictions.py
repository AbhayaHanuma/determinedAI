import numpy as np
import pandas as pd

from determined.experimental import client
from determined.keras import load_model_from_checkpoint_path
from fetch import load_training_data


def get_chpt_preds():
    output = {}
    start_id = 20 #Give the respective ID of the experiment
    for i in range(3,13):#3 to 12 windows
        checkpoint = client.get_experiment(i+start_id).top_checkpoint()
        path = checkpoint.download()
        model = load_model_from_checkpoint_path(path)

        _, _, scaler, _, last_win_data= load_training_data(window=i,val=0) #Getting last window data to get the prediction.

        predict_norm = model.predict(last_win_data)

        predict_denorm = scaler.inverse_transform(np.hstack((last_win_data[-1,-1:,:-1],predict_norm)))
        output[str(i)] = [predict_denorm[-1][-1]]

    output_df = pd.DataFrame(output)
    output_df.to_csv('output.csv') #testing.csv

if __name__=='__main__':
    get_chpt_preds()
