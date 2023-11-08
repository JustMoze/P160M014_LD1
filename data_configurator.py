import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn.impute import SimpleImputer
import numpy as np
from strenum import StrEnum

class ChangeStrategy(StrEnum):
    mean = 'mean'
    median = 'median'
    most_frequent = 'most_frequent'
    constant = 'constant'

class DataConfigurator:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def replace_numeric_nan(self, strategy):
        """
    Replace missing (NaN) values with the most common value in numeric columns.

    This function iterates through the columns of a given DataFrame and identifies
    numeric columns with missing values (NaN). For each such column, it replaces
    the missing values with the most common value found in that column, assuming
    all other non-missing values as the reference.

    Parameters:
        - strategy (str): The imputation strategy to use, which can be one of the following:
            - 'mean': Replace NaN values with the mean of the column.
            - 'median': Replace NaN values with the median of the column.
            - 'most_frequent': Replace NaN values with the most frequent value in the column.
            - 'constant': Replace NaN values with a specified constant value.

    Returns:
        None

    Example Usage:
        data_configurator.replace_numeric_nan(ChangeStrategy.mean)

    Args:
        - strategy (str): The imputation strategy to use.

    """
        nan_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

        print(f"Before data frame configuration: \n{self.dataframe.isna().sum()}")
        for column_name in self.dataframe.columns:
            # Check if column contains numeric values
            if is_numeric_dtype(self.dataframe[column_name]):
                self.dataframe[column_name] = nan_imputer.fit_transform(self.dataframe[[column_name]]) 
        print(f"After data frame configuration: \n{self.dataframe.isna().sum()}")

    def replace_non_numeric_to_most_frequent(self):
        """
    Replace missing non-numeric (categorical) values with the most frequent value.

    This function iterates through the columns of a given DataFrame and identifies
    columns with non-numeric (categorical) data types containing missing values (NaN).
    For each such column, it replaces the missing values with the most frequent value
    found in that column.

    Returns:
        None

    Example Usage:
        data_configurator.replace_non_numeric_to_most_frequent()
        """
        
        for column_name in self.dataframe.columns:
            if is_string_dtype(self.dataframe[column_name]):
                self.dataframe[column_name].fillna(self.dataframe[column_name].mode()[0], inplace=True)

    def drop_dublicates(self):
        """
        Deletes all duplicate rows from the data frame 

        Returns: 
            None

        Example Usage:
            data_configurator.drop_dublicates()
        """
        self.dataframe.drop_duplicates(keep=False,inplace=True)

    def convert_to_list_of_objects(self, object_type, column_number_list):
        """
        Converts a DataFrame to a list of objects of a specified type using column indices.
    
        This function takes a DataFrame, an object type, and a list of column indices as input. It iterates through the rows of the DataFrame and creates objects of the specified type using the data from the specified columns based on their indices. Each row in the DataFrame corresponds to an object in the resulting list.
    
        Parameters:
            object_type (type): The type of objects to be created from the DataFrame rows.
            column_number_list (list): A list of column indices that correspond to the object's constructor parameters.
    
        Returns:
            list: A list of objects of the specified type.
    
        Example Usage:
            # Create a DataFrame from a CSV file
            df = pd.read_csv('data.csv')
    
            # Define the object type
            class MyObject:
                def __init__(self, attribute1, attribute2):
                    self.attribute1 = attribute1
                    self.attribute2 = attribute2
    
            # Define the column indices that correspond to the object's constructor parameters
            column_indices = [1, 2]
    
            # Convert the DataFrame to a list of MyObject objects using column indices
            object_list = data_configurator.convert_to_list_of_objects(MyObject, column_indices)
        """
        objects = []
        for _, row in self.dataframe.iterrows():
            object_data = [row.iloc[i] for i in column_number_list]
            obj = object_type(*object_data)
            objects.append(obj)
        return objects
