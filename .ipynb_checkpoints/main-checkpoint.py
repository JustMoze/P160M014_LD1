import pandas as pd

creditRiskDF = pd.read_csv('credit_risk.csv', index_col="Id")
creditRiskDF.info()

# Dropping duplicate values 
creditRiskDF.drop_duplicates(inplace=True)
creditRiskDF = creditRiskDF.reset_index(drop=True)
creditRiskDF.shape

creditRiskDF.describe