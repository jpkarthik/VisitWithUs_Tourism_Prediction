import streamlit as st
import pandas as pd
import joblib
import os
import logging
from huggingface_hub import login,hf_hub_download
from xgboost import XGBClassifier
#from google.colab import userdata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
cache_dir = "/tmp/hf_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

try:
  hf_token = os.getenv("HUGGINGFACE_TOKEN")

  if hf_token:
    login(token=hf_token)
    logger.info("Successfully logged in to Hugging Face")
  else:
    logger.error("Hugging face token not found")
    st.error("Huggingface token not found")
except Exception as ex:
  logger.error(f"Failed to login to Hugging face: {ex} ")
  st.write(f"Failed to login to Hugging face: {ex} ")

try:
  os.makedirs(cache_dir, exist_ok=True)
  logger.info(f"Created cache directory {cache_dir}")
except Exception as ex:
  logger.error(f"Failed to create cache directory {cache_dir}: {ex}")
  st.error(f"Failed to create cache directory {cache_dir}: {ex}")


st.title("Visit with Us: Tourism Package Prediction")
st.write("Enter the Customer details to predict the likehood of purchasing the tourism packages")


if 'predictor' not in st.session_state:
  st.session_state.predictor = None
  st.session_state.model_loaded = False

class PredictorTourism:

  def __init__(self):
    self.Subfolders = 'Master'
    self.repoID = 'jpkarthikeyan/Tourism_Prediction_Model'
    self.model = None
    self.best_threshold = 0.0

  def Load_Model(self):
    try:
      logger.info("Loading best model")
      model_path = hf_hub_download(
          repo_id = self.repoID,filename = f'Model_Dump_JOBLIB/BestModel_XGBoostingClassifier.joblib',
          repo_type = 'model')
      threshold_path = hf_hub_download(
          repo_id = self.repoID, filename=f'Model_Dump_JOBLIB/best_threshold.txt',
          repo_type='model')

      logger.info(f"Model path: {model_path}")
      logger.info(f"Threshold path:  {threshold_path}")

      self.model = joblib.load(model_path)
      with open(threshold_path,'r') as f:
        self.best_threshold = float(f.read())
      st.success("Model and threshold loaded successfully")
      return True

    except Exception as ex:
      st.error(f'Exception: {ex}')
      return False


  def Predict(self, data):
    try:
      logger.info(f"Input Data: {data}")
      df= pd.DataFrame([data])
      logger.info(f"Data shape: {df.shape}")
      logger.info(f"Dataframe columns: {df.columns.tolist()}")
      prob = self.model.predict_proba(df)[:,1]
      prediction = int(prob >= self.best_threshold)
      return prediction

    except Exception as ex:
      logger.error(f"Exception in predict: {ex}", exc_info=True)
      st.error(f"Exception Prediction: {ex}")
      return ex


if not st.session_state.model_loaded:
  st.session_state.predictor = PredictorTourism()
  st.session_state.model_loaded = st.session_state.predictor.Load_Model()

with st.form("customer_form"):
  st.header("Customer Details")
  col1, col2,col3 = st.columns(3)

  with col1:

    age = st.number_input("Age", min_value=18, max_value=100, value=41)
    gender = st.selectbox('Gender',['Male','Female'])
    MaritalStatus = st.selectbox('MaritalStatus',['Married','Unmarried','Single','Divorced'])
    Occupation = st.selectbox('Occupation',['Free Lancer','Salaried','Small Business','Large Business'])
    Designation = st.selectbox('Designation',['AVP','Manager','Executive','Senior Manager','VP'])
    MonthlyIncome = st.number_input('MonthlyIncome',min_value=0, max_value=1000000,value=20999)

  with col2:

    typeofcontact = st.selectbox("TypeofContact",['Self Enquiry','Company Invited'])
    citytier = st.selectbox('citytier',[1,2,3], index=2)
    DurationOfPitch = st.number_input('DurationOfPitch', min_value=1, max_value=60, value=6)
    ProductPitched = st.selectbox('ProductPitched',['Deluxe','Basic','Standard','Super Deluxe','King'])
    PreferredPropertyStar = st.selectbox("'PreferredPropertyStar",[3,2,1])
    NumberOfTrips = st.number_input('NumberOfTrips',min_value=0, max_value=30, value=1)


  with col3:
    NumberOfPersonVisiting = st.number_input('NumberOfPersonVisiting',min_value=1,max_value=10,value=3)
    NumberOfFollowups = st.number_input('NumberOfFollowups',min_value=0,max_value=10, value=3)
    NumberOfChildrenVisiting= st.number_input('NumberOfChildrenVisiting',min_value=0,max_value=5,value=0)
    Passport= st.selectbox('Passport',['Yes','No'],format_func=lambda x:"Yes" if x=="Yes" else "No")
    Owncar= st.selectbox('OwnCar',['Yes','No'],format_func=lambda x:"Yes" if x=="Yes" else "No")
    PitchSatisfactionScore= st.number_input('PitchSatisfactionScore',min_value=1,max_value=5,value=3)


  submitted = st.form_submit_button("Predict")

if submitted:
  input_data = {
      'Age':age,
      'TypeofContact':typeofcontact,
      'CityTier':citytier,
      'DurationOfPitch':DurationOfPitch,
      'Occupation':Occupation,
      'Gender':gender,
      'NumberOfPersonVisiting':NumberOfPersonVisiting,
      'NumberOfFollowups':NumberOfFollowups,
      'ProductPitched':ProductPitched,
      'PreferredPropertyStar':PreferredPropertyStar,
      'MaritalStatus':MaritalStatus,
      'NumberOfTrips':NumberOfTrips,
      'Passport':1 if Passport =="Yes" else 0,
      'OwnCar':1 if Owncar =="Yes" else 0,
      'PitchSatisfactionScore':PitchSatisfactionScore,
      'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
      'Designation':Designation,
      'MonthlyIncome':MonthlyIncome

  }


  if st.session_state.predictor:
    result = st.session_state.predictor.Predict(input_data)

    if result is not None:
      st.subheader(f"Prediction Result is {result}")
      st.write(f"Likely to purchase" if result ==1 else "Unlikely to purchase")
    else:
      st.write(result)
      st.error("Error in prediction")
  else:
    st.error("Models are not loaded, please ensure the model and threshold are available on Hugging face")

