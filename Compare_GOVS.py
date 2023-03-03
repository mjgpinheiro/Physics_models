# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data from different sources
data1 = pd.read_csv('source1.csv')
data2 = pd.read_excel('source2.xlsx')
data3 = pd.read_json('source3.json')

# clean and preprocess the data
data1.drop_duplicates(inplace=True)
data2.dropna(inplace=True)
data3.fillna(0, inplace=True)

# merge the data into a single DataFrame
merged_data = pd.merge(data1, data2, on='country', how='inner')
merged_data = pd.merge(merged_data, data3, on='country', how='inner')

# define the indicators of interest
indicators = ['GDP per capita', 'Life expectancy', 'Education index']

# create a scatter plot to compare government performance across indicators
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for i, indicator in enumerate(indicators):
    axs[i].scatter(merged_data[indicator], merged_data['government effectiveness'])
    axs[i].set_xlabel(indicator)
    axs[i].set_ylabel('Government effectiveness')
plt.show()
