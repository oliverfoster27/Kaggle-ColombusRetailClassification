# Kaggle-ColombusRetailClassification
 Tracking progress of Kaggle competition classifying online retail purchases for Columbus

### To Run
    1) Configure pipeline architecture in gridsearch.py
    2) Run python gridsearch.py
    3) Track output in console
    4) Take optimal hyperparameters from gridsearch console output to configure architecture in pipeline.py
    5) Run python pipeline.py & take output CSV to submit to kaggle
    
### Relevant Contents
| File                    | Description                                                                                               |
|-------------------------|-----------------------------------------------------------------------------------------------------------|
| gridsearch.py           | Script to configure & run to find optimal parameters for a given pipeline architecture & scoring function |
| pipeline.py             | Script to put the ideal architecture & hyperparameters & generate the submission outputs                  |
| ColombiaDNN_Colab.ipynb | Deep Neural Network attempt run on Google Colab                                                           |