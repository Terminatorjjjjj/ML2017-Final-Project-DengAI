# ML2017 Final Project

## DengAI: Predicting Disease Spread
[Competition Site](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)



## Directory Structure
**ML2017/final/**
-	**src/**
	-	**data/**: Raw trainging and testing data cs files.
	-	**arc/**: Random forest models with lagging labels.
	-	**rfr/**: Random forest models without lagging labels.
	-	**rnn/**: RNN models.
	-	**training/**: Training code.
	-	**ensmeble.sh**: Shell script for reproduction.
	-	**arcanin_rf.py**: Testing code for random forest models with lagging labels. (preprocessing included)
	-	**merge_test.py**: Testing code for random forest models without lagging labels and final ensemble. (preprocessing included)
	-	**rnn2221.py**: Testing code for RNN ensmeble models. (preprocessing included)
-	**Report.pdf**
-	**requirements.txt**
-	**README.md**


## Reproduce Prediction
1. Under **ML2017/final/**, install required *python3.6* packages with **requirements.txt**.
2. Change directory to **src/**.
3. Make sure shell script **ensmeble.sh** is executable.
4. Execute **ensemble.sh**.
5. The final prediction file is generated under **src/** and is named **ensemble_result.csv**. In addition, two intermediate csv files **arc.csv** and **rnn2221.csv** will be generated under the same directory.
