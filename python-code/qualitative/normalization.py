import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def transform(values):
	#Normalized Data
	# print(min(values), max(values))
	# 0.0006863353773951528 0.015413906425237656
	min_=0.0006863353773951528
	max_=0.015413906425237656
	
	normalized = (values-min_)/(max_-min_)

	return normalized

def main():
	filename = "att_clusters_multimodal/clusters_multimodal_correct.csv"
	csv_data = pd.read_csv(filename, names=['0', '1', '2', '3', '4', '5'])

	weights1 = csv_data.values[:,0]
	weights2 = csv_data.values[:,2]
	weights3 = csv_data.values[:,3]
	weights4 = csv_data.values[:,4]
	weights5 = csv_data.values[:,5]
	normalized_weights1 = transform(weights1)
	normalized_weights1 = np.where(normalized_weights1>1, 1, normalized_weights1)
	normalized_weights1 = np.where(normalized_weights1<0, 0, normalized_weights1)
	normalized_weights2 = transform(weights2)
	normalized_weights2 = np.where(normalized_weights2>1, 1, normalized_weights2)
	normalized_weights2 = np.where(normalized_weights2<0, 0, normalized_weights2)
	normalized_weights3 = transform(weights3)
	normalized_weights3 = np.where(normalized_weights3>1, 1, normalized_weights3)
	normalized_weights3 = np.where(normalized_weights3<0, 0, normalized_weights3)
	normalized_weights4 = transform(weights4)q
	normalized_weights4 = np.where(normalized_weights4>1, 1, normalized_weights4)
	normalized_weights4 = np.where(normalized_weights4<0, 0, normalized_weights4)
	normalized_weights5 = transform(weights5)
	normalized_weights5 = np.where(normalized_weights5>1, 1, normalized_weights5)
	normalized_weights5 = np.where(normalized_weights5<0, 0, normalized_weights5)


	# create a dataframe to save
	indexes = np.arange(150)
	headers = ['timestep','att_1', 'att_2','att_3', 'att_4', 'att_5']
	data = {"timestep": indexes, "att_1": normalized_weights1, "att_2": normalized_weights2, 
	"att_3": normalized_weights3, "att_4": normalized_weights4, "att_5": normalized_weights5}
	dataframe = pd.DataFrame(data, columns=['timestep', 'att_1','att_2',
		'att_3','att_4','att_5'])

	dataframe.to_csv("att_clusters_multimodal/clusters_multimodal_correct_normalized.csv")

	

if __name__ == '__main__':
	main()