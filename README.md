Stock prediction using LSTM Machine Learning

Many-to-One LSTM taking price and volume for each minute as inputs and a single heuristic output (measured with future prices). Using Keras library (a wrapper for tensorflow).

* Models Folder - Generated models from the script

* RawData Folder - Stock data in csv format

* Run.py
  * Parses data from RawData folder into a list of times, tickers, opens, and volumes during trading hours
  * Split data into train and test data
  * Generate models for each minute based on data from earlier in day
  * Test models with test data

* Scratch Folder - Contains files used to test previous stock algorithm ideas
