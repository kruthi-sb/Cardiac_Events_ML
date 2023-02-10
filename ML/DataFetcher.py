#create a class for fetchinf the data from the csv file and splitting it into training and testing sets.
class DataFetcher:
    
    def __init__(self, file_path):
        import sklearn as sk, pandas as pd, numpy as np
        self.file_path = file_path
        self.full_dset = pd.read_csv(self.file_path)
        self.full_dset.drop(["index"], axis = 1, inplace = True)
        self.y_s = self.full_dset["output"] #y holds target variable as a pandas Series.It has 300 rows.
        self.y_df = pd.DataFrame(self.y_s) 
        self.X_df = self.full_dset.drop("output", axis=1, inplace = False)# X holds only the 13 feature columns by droping output column.
        self.features = self.X_df.columns #features is a list of 13 feature names. 
        self.target_names = ["diseased", "non_diseased"]
        self.X = self.X_df.values #X is a numpy array of shape (300,13)
        self.y = self.y_df.values #y is a numpy array of shape (300,1)
        from sklearn.utils import column_or_1d
        self.y = column_or_1d(self.y, warn=False) #y is a numpy array of shape (300,)
        #splitting the dataset into training, validation and testing sets.
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


        

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test
    
    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_features(self):
        return self.features

    def get_target_names(self):
        return self.target_names

    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y





print("\n------class run complete------\n")