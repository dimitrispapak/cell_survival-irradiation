# cell_survival-irradiation
An ML application that predicts the cell survival depending on the features of the radiation it receives

To run the program you need to pip install the following dependencies:
1. joblib==0.13.2
-matplotlib==3.1.1
-numpy==1.16.4
-seaborn==0.9.0
-pandas==0.24.2
-catboost==0.15.2
-shap==0.29.3
-scikit_learn==0.21.2

The application is loading two csv files, which are the datasets that contains data from irradiation studies
One of the files holds data that contain the coefficients of the quadratic model, in the code it called pide.csv and it's  the Particle Irradiation Data Ensemble (PIDE) dataset provided by the GSII in Germany.
The other dataset (data_rbe) is pretty similar to the pide.csv, but the irradiation effect on the cells is expressed with RBE quantity.

Workflow:
- Run the catboost_model.py
  this will create model that will be saved and some performance plots in the plots folder
- Similaliry, run the support_vector_regression.py
 - You can either run the voting_regression.py, or you can use the parameter_grid.py script to find the best suited parameters that are needed for the regressors in the voting_reggresion
- Now, you can run main.py and input the features of the radiation that are requested
- If you don't have some of them just press enter, and the program will impute the value from the existing dataset
