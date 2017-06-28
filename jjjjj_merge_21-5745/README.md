# ML2017 Final Project

## DengAI: Predicting Disease Spread
[Competition Site](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)


## Directory Structure
**ML2017/final/**
	-	**src/**
		+	**data/**: Raw trainging and testing data.
		+	**arc/**: Random forest models with lagging labels.
		+	**rfr/**: Random forest models without lagging labels.
		+	**rnn/**: RNN models.
		+	**ensmeble.sh**: Shell script for reproduction.
		+	**arcanin_rf.py**: Testing code for random forest models with lagging labels.
		+	**merge_test.py**: Testing code for random forest models without lagging labels.
		+	**rnn2221.py**: Testing code for RNN ensmeble models.
	-	**Report.pdf**
	-	**requirements.txt**


## Reproduce Prediction
1. Under ML2017/final/, install required packages with **requirements.txt**.
2. Change directory to src/.
3. Make sure shell script **ensmeble.sh** is executable.
4. Run **ensemble.sh**.
5. The prediction file is generated under src/ and is named **ensemble_result.csv**.