import os
import traceback
import inspect
from huggingface_hub import HfApi, create_repo,login,hf_hub_download

class DataRegistration:
  def __init__(self,base_path,hf_token=None):
    self.repoID = 'jpkarthikeyan/Tourism-visit-with-us-dataset'
    self.Subfolders = os.path.join(base_path,'Data')
    self.folder_Master = base_path
    self.folder_data = os.path.join(base_path,"Data")
    self.hf_token = hf_token
    os.makedirs(self.folder_data, exist_ok=True)

  def HFCreateRepo(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      create_repo(repo_id=self.repoID,
                  private=False,
                  repo_type='dataset',
                  exist_ok=True)
      print(f"Repo {self.repoID} created")
      return True
    except Exception as ex:
      if hasattr(ex,'response') and ex.response.status_code == 409:
        print(f"Repo {self.repoID} already exists")
        return True
      else:
        print(f"Exception {ex}")
        traceback.print_exc()
        return False
    finally:
      print("-"*100)


  def UploadingSourceData(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      source_data_file = os.path.join(self.folder_data,'tourism.csv')
      if not os.path.exists(source_data_file):
        raise FileNotFoundError(f"File {source_data_file} not found")
      api = HfApi()
      api.upload_file(
          path_or_fileobj = source_data_file,
          path_in_repo = 'Master/Data/tourism.csv',
          repo_id = self.repoID,
          repo_type='dataset',
          token=self.hf_token)
      print(f"Source data tourism.csv uploaded into {self.repoID}")
      return True

    except Exception as ex:
       print(f"Exception {ex}")
       traceback.print_exc()
       return False
    finally:
      print("-"*100)

  def ToRunPipeline(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    if not self.HFCreateRepo():
      print("Exception occured while creating the repo in DataRegistration.py")
      return False
    else:
      if not self.UploadingSourceData():
        print("Exception occured while uploading the source data into HF dataset")
        return False
      else:
        print("FunctionToRunPipeline completed")
        return True
    

