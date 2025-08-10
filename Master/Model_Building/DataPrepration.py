import os
import pandas as pd
import inspect
import traceback
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo, login, hf_hub_download

class DataPrepration:
  def __init__(self,base_path, hf_token=None):
    self.repoID = 'jpkarthikeyan/Tourism-visit-with-us-dataset'
    self.Subfolders = os.path.join(base_path, 'Data')
    self.hf_token = hf_token
    

  def LoadDatasetFromHF(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      loading_dataset = load_dataset(path=self.repoID,
                  data_files={'train':'Master/Data/tourism.csv'},
                  token=self.hf_token)
      df_dataset = pd.DataFrame(loading_dataset['train'])
      print(f'Shape {df_dataset.shape}')

      if 'Unnamed: 0' in df_dataset.columns:
        df_dataset = df_dataset.drop(['Unnamed: 0'],axis=1)

      print(f"Dataset loaded from {self.repoID}/Master/Data/")
      print(f"Shape of the Original Dataset: {df_dataset.shape}")
      return df_dataset
    except Exception as ex:
      print(f"Exception in LoadDatasetFromHF {ex}")
      traceback.print_exc()
      return None
    finally:
      print('-'*50)


  def TrainTestSplit(self,df_dataset):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      print(f"Value Count {df_dataset['ProdTaken'].value_counts()}")

      df_train,df_test = train_test_split(df_dataset,
                              test_size=0.2,random_state=42,
                              stratify=df_dataset['ProdTaken'],
                              shuffle=True)
      print(f"Shape of the train dataset: {df_train.shape}")
      print(f"Shape of the test dataset: {df_test.shape}")

      return df_train, df_test
    except Exception as ex:
      print(f'Exception: {ex}')
      print(traceback.print_exc())
      return None, None
    finally:
      print('-'*50)

  def DatasetCleaning(self,df_data):
    try:
      print(f"Function Name {inspect.currentframe().f_code.co_name}")
      df_data['Gender'] = df_data['Gender'].replace('Fe Male', 'Female')

      df_data = df_data.drop_duplicates(subset=['CustomerID'], keep='first').reset_index(drop=True)

      for clmn in df_data.columns:
        if df_data[clmn].dtype in ['int64']:
          #print(f"{clmn} replacing the missing value with median")
          df_data[clmn] = df_data[clmn].fillna(df_data[clmn].median())
        else:
          #print(f"{clmn} replacing the missing value with mode")
          df_data[clmn] = df_data[clmn].fillna(df_data[clmn].mode()[0])

      df_data = df_data.drop(['CustomerID'], axis=1)

      numerical_column = df_data.select_dtypes(include=['int64'])

      for num_col in numerical_column:
        Q1 = df_data[num_col].quantile(0.25)
        Q3 = df_data[num_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        #df_data[num_col] = df_data[num_col].clip(lower,upper)

      return df_data

    except Exception as ex:
      print(f"Exception {ex}")
      print(traceback.print_exc())
      return None
    finally:
      print('-'*50)
    

  def UploadIntoHF(self,df,drive_path,file_name):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      file_path = os.path.join(drive_path,file_name)
      df.to_csv(file_path,index=False)

      api = HfApi(token=self.hf_token)
      api.upload_file(
          path_or_fileobj = file_path,
          path_in_repo= f"Master/Data/{file_name}",
          repo_id = self.repoID,
          repo_type='dataset',
          token=self.hf_token)
      print(f"Source data {file_name} uploaded into {self.repoID}")
      return True
    except Exception as ex:
      print(f"Exception: {ex}")
      traceback.print_exc()
      return False
    finally:
      print('-'*50)

  def ToRunPipeline(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    df_dataset = self.LoadDatasetFromHF()
    if df_dataset is None:
      print("dataset loading exception")
      return False
    else:
      df_train, df_test = self.TrainTestSplit(df_dataset)
      if df_train is None or df_test is None:
        print('Dataset train test split exception')
        return False
      else:
        df_train_cleaned = self.DatasetCleaning(df_train)
        df_test_cleaned = self.DatasetCleaning(df_test)
        if df_train_cleaned is None or df_test_cleaned is None:
          print('Dataset cleaning failed')
          return False
        else:
          result_train =self.UploadIntoHF(df_train_cleaned,
                            self.Subfolders,'train.csv')
          result_test = self.UploadIntoHF(df_test_cleaned,
                                  self.Subfolders,'test.csv')
          if not result_train or not result_test:
            print('Splitted dataset upload into HF Exception')
            return False
          else:
            print("dataset downloaded from HF, cleaned, " \
            "splitted into train and test dataset and " \
            "uploaded back into HF dataset")
            return True       