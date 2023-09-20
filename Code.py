#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import requests
import zipfile
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pyodbc
import json


# In[2]:


data_NHTSA = "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip"
data_FUEL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"


# In[55]:


# Please, change this variables when you run the script.
save_path = "please_enter_your_save_path/data.zip" 
extract_path = "please_enter_your_extract_path"


# In[4]:


# I - Data Acquistion
def download_and_extract_zip(url, save_path, extract_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded the ZIP file to {save_path}")
        
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted the contents to {extract_path}")
        
        os.remove(save_path)
        print(f"Removed the downloaded ZIP file: {save_path}")
    else:
        print("Failed to download the dataset.")


# In[8]:


def read_dataset(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension not in ('csv', 'txt'):
        raise ValueError("Unsupported file format. Only CSV and TXT files are supported.")
    if file_extension == 'csv':
        df = pd.read_csv(file_path, sep=',', engine='python')
        df = df.apply(lambda x: x.replace('', np.nan), axis=0)
    else: 
        lines = []
        with open(file_path, 'r') as file:
            for line in file:
                fields = line.strip().split('\t')
                if fields[0].isdigit():
                    lines.append(fields)
        df = pd.DataFrame(lines)
        df = df.replace('', np.nan)
    return df


# In[31]:


df_vehicles = read_dataset("/your_path/")
df_vehicles.head()
df_vehicles.dtypes


# In[10]:


df_CMPL = read_dataset("/your_path")
df_CMPL.head()


# In[11]:


def rename_columns(df, column_names):
    df.columns = column_names
    return df


# In[12]:


new_column_names = [
    'CMPLID',
    'ODINO',
    'MFR_NAME',
    'MAKETXT',
    'MODELTXT',
    'YEARTXT',
    'CRASH',
    'FAILDATE',
    'FIRE',
    'INJURED',
    'DEATHS',
    'COMPDESC',
    'CITY',
    'STATE',
    'VIN',
    'DATEA',
    'LDATE',
    'MILES',
    'OCCURENCES',
    'CDESCR',
    'CMPL_TYPE',
    'POLICE_RPT_YN',
    'PURCH_DT',
    'ORIG_OWNER_YN',
    'ANTI_BRAKES_YN',
    'CRUISE_CONT_YN',
    'NUM_CYLS',
    'DRIVE_TRAIN',
    'FUEL_SYS',
    'FUEL_TYPE',
    'TRANS_TYPE',
    'VEH_SPEED',
    'DOT',
    'TIRE_SIZE',
    'LOC_OF_TIRE',
    'TIRE_FAIL_TYPE',
    'ORIG_EQUIP_YN',
    'MANUF_DT',
    'SEAT_TYPE',
    'RESTRAINT_TYPE',
    'DEALER_NAME',
    'DEALER_TEL',
    'DEALER_CITY',
    'DEALER_STATE',
    'DEALER_ZIP',
    'PROD_TYPE',
    'REPAIRED_YN',
    'MEDICAL_ATTN',
    'VEHICLES_TOWED_YN'
]


# In[13]:


df_CMPL = rename_columns(df_CMPL, new_column_names)


# In[14]:


df_CMPL.head()


# In[15]:


df_FUEL = read_dataset("/your_path/alt_fuel_stations.csv")
df_FUEL.head()


# In[56]:


# II - Data Preprocessing
def preprocess_dataframe(df, flag_standardize=False, threshold=80):
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove variables with the same value for all rows
    df = df.loc[:, df.nunique() != 1]

    # Remove variables with more than 'threshold' percentage of np.nan
    nan_percentage = (df.isna().sum() / len(df)) * 100
    columns_to_remove = nan_percentage[nan_percentage > threshold].index
    df = df.drop(columns=columns_to_remove)

    # Separate numerical and non-numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    # Fill in missing values
    for col in df.columns:
        if col in numerical_columns:
            # Fill missing values in numerical columns with median SINCE it's more robust than mean
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # Fill missing values in non-numeric columns with majority vote
            df[col].fillna(df[col].mode().iloc[0], inplace=True)

    # Remove outliers in numerical columns using IQR method
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

    # Standardize numerical columns if flag_standardize is True
    if flag_standardize:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df


# In[64]:


df_processed = preprocess_dataframe(df_FUEL,False,80)


# In[59]:


# III - Data Transformation
def convert_dataframe(dataframe, output_path, to_format):
    try:
        if to_format == 'csv':
            dataframe.to_csv(output_path, index=False)
            print(f"Data converted to CSV and saved to {output_path}")
        elif to_format == 'json':
            dataframe.to_json(output_path, orient='records')
            print(f"Data converted to JSON and saved to {output_path}")
        else:
            print("Invalid target format. Use 'csv' or 'json'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# In[ ]:


convert_dataframe(df_FUEL, "/your_path/data.json", 'json')


# In[65]:


#IV - Data Loading

# connection parameters
server = 'serve_name'
database = 'database_name'
username = 'username'
password = 'password'
driver = '{ODBC Driver 17 for SQL Server}'  # I assume that i want to use this driver

# Define the connection string
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Create a database connection within a 'with' statement
with pyodbc.connect(connection_string) as conn:
    cursor = conn.cursor()

    # Load data from a JSON file or CSV file 
    file_path = 'your_data.json'  
    file_extension = file_path.split('.')[-1]

    # Load JSON or CSV data into a DataFrame
    if file_extension.lower() == 'json':
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            df = pd.DataFrame(data)
    elif file_extension.lower() == 'csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Only JSON and CSV are supported.")

    column_names = df.columns.tolist()

    # creating a table 
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS YourTableName (
        {', '.join([f'{col} VARCHAR(255)' for col in column_names])}
    )
    """
    cursor.execute(create_table_sql)

    # loading data into the table
    load_data_sql = f"""
    INSERT INTO YourTableName ({', '.join(column_names)})
    VALUES ({', '.join(['?'] * len(column_names))})
    """
    # fill in the table
    for index, row in df.iterrows():
        cursor.execute(load_data_sql, *row)
        conn.commit()

    print("Data loaded WITH SUCESS!.")


# In[ ]:




