import pandas as pd

creditRiskDF = pd.read_csv('credit_risk.csv', index_col="Id")
creditRiskDF.info()