import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
base_path = os.path.join(base_path,'Master')
print(f'Base path {base_path}')
sys.path.append(os.path.join(base_path,'Model_Building'))
print(f'System Path {sys.path.append(os.path.join(base_path,'Model_Building'))}')
from DataRegistration import DataRegistration


data_dir = os.path.join(base_path, 'Data')
model_dir = os.path.join(base_path,'Model_Dump_JOBLIB')
#job = ['register','prepare']
#job = 'prepare'

parser = argparse.ArgumentParser(description='Run a specific job in the pipeline')
parser.add_argument('--job', type=str, required=True,
                    choices=['register','prepare','modelbuilding'],
                    help='Job To execute register or prepare')
args = parser.parse_args()

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

hf_token = os.getenv('HF_TOKEN')
if not hf_token:
  raise ValueError("HF_TOKEN not found in .env file")

if args.job == 'register':
  data_reg = DataRegistration(base_path, hf_token)
  if not data_reg.ToRunPipeline():
    sys.exit(1)
elif args.job == 'prepare':
  from DataPrepration import DataPrepration
  obj_data_prep = DataPrepration(base_path,hf_token)
  if not obj_data_prep.ToRunPipeline():
    sys.exit(1)
elif args.job == 'modelbuilding':
  from BuildingModels import BuildingModels
  ObjBuildModel = BuildingModels(base_path,hf_token)
  if not ObjBuildModel.ToRunPipeline():
    sys.exit(1)