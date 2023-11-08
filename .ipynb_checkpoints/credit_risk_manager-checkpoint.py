from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from loan_applicant import LoanApplicantProfile
class CreditRiskManager: 
    def __init__(self, applicants: list[LoanApplicantProfile]):
        self.applicants = applicants
        self._applicantsDF = self._configure_dataframe()

    def _configure_dataframe(self):
        """
        Configures the input data from the list of LoanApplicantProfile objects into a DataFrame for analysis.
        
        Returns:
            pd.DataFrame: A configured DataFrame with the necessary data for analysis.
        """
        loanDF = pd.DataFrame([vars(applicant) for applicant in self.applicants])
        loanDF.drop(["id"], axis=1, inplace=True)
        labelEncoder = LabelEncoder()
        
        loanApplicants = loanDF.select_dtypes(include = "object").columns
        for loanApplicant in loanApplicants:
            loanDF[loanApplicant] = labelEncoder.fit_transform(loanDF[loanApplicant].astype(str))
        
        return loanDF

    def plot_age_distribution(self):
        """
        Plots the distribution of ages of loan applicants.
    
        Returns:
            None
        """
        ages = [applicant.age for applicant in self.applicants]
        plt.hist(ages, bins=20, edgecolor='k')
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

    def plot_home_type_distribution(self):
        """
        Plots the distribution of home types among loan applicants.
    
        Returns:
            None
        """
        homes = [applicant.home for applicant in self.applicants]
        homesDF = pd.DataFrame(homes, columns=["home"])
        sns.countplot(homesDF, x="home")
        plt.title('Home', fontsize = 20)
        plt.xticks(fontsize = 12) 
        plt.show()

    def plot_intent_distribution(self):
        """
        Plots the distribution of loan intents among loan applicants.
    
        Returns:
            None
        """
        intents = [applicant.intent for applicant in self.applicants]
        intentsDF = pd.DataFrame(intents, columns=["intent"])
        sns.countplot(intentsDF, x="intent")
        plt.title('Intent', fontsize = 20)
        plt.xticks(fontsize = 8) 
        plt.show()

    def show_correlation_between_applicant_parameters(self):
        """
        Displays a heatmap of the correlation between applicant parameters.
        This method computes the correlation between different applicant parameters and displays it as a heatmap.
    
        Returns:
            None
        """
        f, ax = plt.subplots(figsize=(6, 6))
        print(f"self._applicantsDF.corr() \n{self._applicantsDF.corr()}")
        sns.heatmap(self._applicantsDF.corr(), annot=True, fmt='.2f', cmap='RdYlGn',annot_kws={'size': 7}, ax=ax)
        plt.show()

    def plot_applicant_parameter_importance(self):
        """
        Plots the feature importance for predicting loan status.
        This method uses a RandomForestClassifier model to determine the feature importance in predicting the loan status.
    
        Returns:
            None
        """
        y = self._applicantsDF['status']
        X = self._applicantsDF.drop('status', axis='columns')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 77)

        model = RandomForestClassifier(criterion="gini", min_samples_split=2, random_state=77)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print('Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        print(f"feature_importance_df \n{feature_importance_df}")
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()
