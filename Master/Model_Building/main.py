import os
import sys
from dotenv import load_dotenv

load_dotenv()

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
base_path = os.path.join(base_path,'Master')
print(f'Base path {base_path}')
sys.path.append(os.path.join(base_path,'Model_Building'))
print(f'System Path {sys.path.append(os.path.join(base_path,'Model_Building'))}')
from DataRegistration import DataRegistration
from DataPrepration import DataPrepration
from BuildingModels import BuildingModels

data_dir = os.path.join(base_path, 'Data')
model_dir = os.path.join(base_path,'Model_Dump_JOBLIB')
job = 'register'

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

hf_token = os.getenv('HF_TOKEN')
if not hf_token:
  raise ValueError("HF_TOKEN not found in .env file")

if job == 'register':
  data_reg = DataRegistration(base_path, hf_token)
  if not data_reg.To_run_Pipeline():
    sys.exit(1)


elif job == 'prepare':
  data_prep = DataPrepration(base_path, hf_token)
  df_dataset = data_prep.LoadDatasetFromHF()
  if df_dataset is not None:
    df_train, df_test = data_prep.TrainTestSplit()
    if df_train is not None and df_test is not None:
      df_train_cleaned = data_prep.CleanData(df_train)
      df_test_cleaned = data_prep.CleanData(df_test)
      data_prep.UploadIntoHF(df_train_cleaned,data_dir,'train.csv')
      data_prep.UploadIntoHF(df_test_cleaned,data_dir,'test.csv')

elif job == 'build':
  model_build = BuildingModels(base_path, hf_token)
  model_build.Load_data_from_HF()
  model_build.Preprocessing_dataset()
  models = model_build.Building_Models()
  if models:
    metrics = model_build.Model_Evaluation()
    model_build.Register_BestModel_HF()
  else:
    print("Unable to build the models")




  


  
