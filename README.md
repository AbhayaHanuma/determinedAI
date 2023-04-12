# AI Sales Astrologer

## Project Objective

The objective of the AI Sales Astrologer project is to create a predictive model for forecasting sales in the apparel industry using a custom multi-head CNN LSTM architecture.

## About the Dataset

The Apparel dataset is a multivariate time series dataset which contains sales data from the year 2014 to 2020 monthly data (i.e 78 data points) as training data, data from mid-2020 to December 2022 (i.e 18 data points) as validation data and we are predicting for the first month of 2023 (i.e 1 data point) which is our testing data. The dataset includes features like Discounts and Demands, and the target variable is Gross sales. And Data preprocessing techniques include scaling and windowing which where included in `fetch.py`.

## Model Architecture

The AI Sales Astrologer model architecture is a custom multi-head CNN LSTM architecture built using Keras and TensorFlow. It comprises of a multihead with 3 Convolutional layers followed by few LSTM and dropout layers. Each of these layers has its own set of hyperparameters which have to be tuned. And this we acheived with the determinedAI.

## How to Run the Experiment


To run the training job, follow the below instructions:

1. Clone the repository to your local machine.
2. Navigate to the project directory and open a command prompt or terminal.
3. Install the required dependencies such as `keras_multi_head` and `scikit-learn` using the pip command, along with the `determined` package.
4. Set up the Determined AI platform on your local machine (see the Determined AI documentation for instructions [here](https://docs.determined.ai/latest/)).
5. Make sure the Determined AI WebUI is up and running to monitor the model training.
6. The dataset is available in the `data` folder of the repository.
7. To run the experiment with the predefined hyperparameters defined in `const.yaml`, use the following command:
```
det -m <master host:port> experiment create -f const.yaml .
```
8. To run for a range of hyperparameters, use the following command (see `adaptive.yaml` for more insights):
```
det -m <master host:port> experiment create -f adaptive.yaml .
```
9. Change the `window` parameter in `const.yaml` and `adaptive.yaml` to run for different window sizes.
10. Once the experiment is done, the metrics can be seen in the Determined AI WebUI on `<master host:port>`.
11. The metric defined is loss, and it can be seen how the loss is changing with an increase in batches.

![Loss changing with increasing batches](/images/loss.png)

12. Once the best trial is identified, load the model from that particular checkpoint to get the final predictions for test data. This can be done using `get_predictions.py`.
13. The predicted values can be saved to a CSV file for further analysis.
