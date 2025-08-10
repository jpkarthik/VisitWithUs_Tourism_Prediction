
import os
import joblib
import inspect
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo, login
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, precision_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class BuildingModels:
  def __init__(self,base_path, hf_token=None):
    self.models = {}
    self.best_model = None
    self.best_score = 0
    self.best_f1_score =0.0
    self.best_model_threshold = 0.0
    self.best_model_name=None
    self.df_train = pd.DataFrame()
    self.df_test = pd.DataFrame()
    self.feature_train = pd.DataFrame()
    self.feature_test = pd.DataFrame()
    self.target_train = pd.Series()
    self.target_test = pd.Series()
    self.base_path = base_path
    self.Subfolders = os.path.join(base_path,'data')
    self.repo_id = 'jpkarthikeyan/Tourism_Prediction_Model'
    self.ds_repo_id = 'jpkarthikeyan/Tourism-visit-with-us-dataset'
    self.repo_type = 'model'
    self.hf_token = hf_token
    self.categorical_columns = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
    self.numerical_columns = ['Age','CityTier','DurationOfPitch','NumberOfPersonVisiting',
                              'NumberOfFollowups','PreferredPropertyStar',
                              'NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar',
                              'NumberOfChildrenVisiting','MonthlyIncome']

    self.pipeline_numerical = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    self.pipeline_onehot = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))
    ])



  def Load_data_from_HF(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      print(f'Loading the train dataset from {self.ds_repo_id}')
      load_train = load_dataset(path=self.ds_repo_id,data_dir=self.Subfolders,
                                data_files={'train':'train.csv'})
      self.df_train = pd.DataFrame(load_train['train'])
      print(f"Shape of the train dataset: {self.df_train.shape}")

      print(f'Loading the test dataset from {self.ds_repo_id}')
      load_test = load_dataset(path=self.ds_repo_id,data_dir=self.Subfolders,
                                data_files={'test':'test.csv'})
      self.df_test = pd.DataFrame(load_test['test'])

      print(f"Shape of the test dataset: {self.df_test.shape}")

    except Exception as ex:
      print(f"Exception: {ex}")
      traceback.print_exc()
      raise

    finally:
      print('-'*50)

  def Preprocessing_dataset(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:

      self.target_train = self.df_train['ProdTaken']
      self.feature_train = self.df_train.drop(['ProdTaken'],axis=1)

      self.target_test = self.df_test['ProdTaken']
      self.feature_test = self.df_test.drop(['ProdTaken'],axis=1)

    except Exception as ex:
      print(f"Exception: {ex}")
      traceback.print_exc()
    finally:
      print('-'*50)

  def Building_Models(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      preprocessor = ColumnTransformer(
          transformers=[
              ('num', self.pipeline_numerical,self.numerical_columns),
              ('onehot', OneHotEncoder(drop='first',handle_unknown='ignore',
                        sparse_output=False),self.categorical_columns)])
      models_params = {
          'DecisionTreeClassifier':{
              'model': DecisionTreeClassifier(class_weight='balanced',random_state=42),
              'params': {'classifier__criterion':['gini','entropy'],
                         'classifier__splitter':['best','random'],
                        'classifier__max_depth':[1],
                         'classifier__min_samples_leaf':[1,2,4],
                         'classifier__min_samples_split':[2,5,10],
                         'classifier__max_features':['sqrt','log2',None]}
          },

          'RandomForestClassifier':{
              'model': RandomForestClassifier(class_weight='balanced',random_state=42),
              'params': { 'classifier__n_estimators':[25,50,75,100],
                          'classifier__criterion':['gini','entropy'],
                          'classifier__max_depth':[5,10,15],
                          'classifier__min_samples_split':[15,20,25],
                          'classifier__min_samples_leaf':[7,10,15],
                          'classifier__max_features':[0.3,0.5,0.6],
                          'classifier__oob_score':[True],
                          'classifier__bootstrap':[True]
                         }
          },

          'BaggingClassifier':{
              'model': BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced',random_state=42)),
              'params':{  'classifier__n_estimators':[10,50,75,100],
                          'classifier__max_samples':[0.3,0.5,0.7,0.9],
                          'classifier__max_features':[0.3,0.5,0.7],
                          'classifier__oob_score':[True],
                          'classifier__estimator__criterion':['gini','entropy'],
                          'classifier__estimator__max_depth':[5,7,9],
                          'classifier__estimator__min_samples_split':[8,10,12],
                          'classifier__estimator__min_samples_leaf':[2,3,5]
                        }
          },

          'AdaBoostingClassifier':{
              'model': AdaBoostClassifier(random_state=42),
              'params':{  'classifier__n_estimators':[50,75,100],
                          'classifier__learning_rate':[0.01,0.05,0.1],
                          'classifier__algorithm':['SAMME','SAMME.R']

                      }
          },

          'GradientBoostingClassifier':{
              'model': GradientBoostingClassifier(random_state=42),
              'params':{
                          'classifier__n_estimators':[50,75,100,125],
                          'classifier__learning_rate':[0.01,0.5,0.1],
                          'classifier__criterion':['friedman_mse','squared_error'],
                          'classifier__max_features':['sqrt','log2'],
                          'classifier__min_samples_leaf':[1,2,4],
                          'classifier__subsample':[0.6,0.7,0.8],
                          'classifier__max_depth':[2,3,4,5]
                        }
          },

          'XGBoostingClassifier':{
              'model':XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
              'params':{'classifier__n_estimators':np.arange(50,100,10),
                        'classifier__max_depth': [3,5,7],
                        'classifier__learning_rate':[0.01,0.1,0.2],
                        'classifier__subsample':[0.6,0.8,1.0],
                        'classifier__colsample_bytree':[0.6,0.8,1.0],
                        'classifier__gamma':[0,1,2],
                        'classifier__reg_alpha':[0,1,2]

                        }
          }

        }

      cv_KFold = KFold(n_splits=3,random_state=42,shuffle=True)

      for model_name, mdl_params in models_params.items():
        print(f'Model {model_name} started')
        pipeline = Pipeline(steps=[
            ('preprocessor',preprocessor),
            ('classifier',mdl_params['model'])
        ])
        random_search = RandomizedSearchCV(pipeline,mdl_params['params'],
                                           n_iter=50,cv=cv_KFold,scoring='f1',
                                           random_state=42,n_jobs=-1,verbose=2)

        random_search.fit(self.feature_train,self.target_train)

        self.models[model_name] = {
            'model':random_search.best_estimator_,
            'best_score': random_search.best_score_,
            'best_params':random_search.best_params_
        }
        joblib.dump(random_search.best_estimator_,f'{self.base_path}/Model_Dump_JOBLIB/{model_name}.joblib')
        print(f'model:{random_search.best_estimator_}')
        print(f'best_score: {random_search.best_score_}')
        print(f'best_params: {random_search.best_params_}')
        print(f'Modle {model_name} completed')
        print('-'*50)

      return self.models
    except Exception as ex:
      print(f"Exception: {ex}")
    finally:
      print('-'*50)


  def Model_Evaluation(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    df_metrics = pd.DataFrame()
    try:
      for mdl_name, mdl_info in self.models.items():
        model = mdl_info['model']
        predict_proability = model.predict_proba(self.feature_test)
        print(f"Predict proability shape {mdl_name} {predict_proability.shape}")
        if predict_proability.shape[1] ==1:
          predict_proability = predict_proability.flatten()
        else:
          predict_proability = predict_proability[:,1]


        prc_precision,prc_recall, prc_threshold = precision_recall_curve(self.target_test,predict_proability)
        prc_f1score = 2*((prc_precision*prc_recall) / (prc_precision+prc_recall+1e-10))

        prc_threshold_idmx = np.argmax(prc_f1score)
        prc_best_threshold = prc_threshold[prc_threshold_idmx]
        print(f'best threshold: {prc_best_threshold}')

        predic_prob_threshold = (predict_proability >= prc_best_threshold).astype(int)
        #predic_prob_threshold = (predict_proability >= 0.5).astype(int)
        accuracy = accuracy_score(self.target_test,predic_prob_threshold)
        precision = precision_score(self.target_test,predic_prob_threshold)
        recall = recall_score(self.target_test,predic_prob_threshold)
        f1score = f1_score(self.target_test,predic_prob_threshold)
        class_report = classification_report(self.target_test,predic_prob_threshold)
        conf_matrix = confusion_matrix(self.target_test,predic_prob_threshold)

        lbl = ['TN', 'FP', 'FN', 'TP']
        cnf_lbl = ['\n{0:0.0f}'.format(cnf_val) for cnf_val in conf_matrix.flatten()]
        cn_percentage = ["\n{0:.2%}".format(item/conf_matrix.flatten().sum()) for item in conf_matrix.flatten()]

        confusion_label = np.asarray([["\n {0:0.0f}".format(item)+"\n{0:.2%}".format(item/conf_matrix.flatten().sum())]
                                for item in conf_matrix.flatten()]).reshape(2,2)

        cnf_label = np.asarray([f'{lbl1} {lbl2} {lbl3}' for lbl1, lbl2, lbl3 in zip(lbl, cnf_lbl,  cn_percentage)]).reshape(2,2)

        plt.figure(figsize = (3,3))
        sns.heatmap(conf_matrix, annot = cnf_label, cmap = 'Spectral', fmt='' )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{mdl_name} confusion matrix')
        plt.tight_layout()
        plt.show()

        df_metrics = pd.concat([df_metrics,pd.DataFrame({'model':[mdl_name],'accuracy':[accuracy],
                                            'precision':[precision], 'recall':[recall],
                                            'f1_score':[f1score]})],ignore_index=True)

        if f1score > self.best_f1_score:
          self.best_f1_score = f1score
          self.best_model_threshold = prc_best_threshold
          self.best_model_name = mdl_name

      best_model = self.models[self.best_model_name]['model']
      if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature':self.feature_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance',ascending=False)
        print('Feature Importance:\n',feature_importance)


      return df_metrics

    except Exception as ex:
      print(f"Exception: {ex}")
    finally:
      print('-'*50)


  def Register_BestModel_HF(self):
    print(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      best_model = self.models[self.best_model_name]['model']
      joblib.dump(best_model,f'{self.base_path}/Model_Dump_JOBLIB/BestModel_{self.best_model_name}.joblib')


      api = HfApi()
      try:
        api.repo_info(repo_id=self.repo_id,repo_type=self.repo_type)
      except RepositoryNotFoundError:
        api.create_repo(repo_id=self.repo_id, repo_type=self.repo_type,private=False)


      print("Uploading the best model into Hugging face")
      api.upload_file(path_or_fileobj = f'{self.base_path}/Model_Dump_JOBLIB/BestModel_{self.best_model_name}.joblib',
                      path_in_repo = f"Model_Dump_JOBLIB/BestModel_{self.best_model_name}.joblib",
                      repo_id=self.repo_id, repo_type=self.repo_type
                      )


      print("Uploading the best threshold text file to HF")
      with open('Master/Model_Dump_JOBLIB/best_threshold.txt','w') as f:
        f.write(str(self.best_model_threshold))
      api.upload_file(path_or_fileobj = f"{self.base_path}/Model_Dump_JOBLIB/best_threshold.txt",
                      path_in_repo = f"Model_Dump_JOBLIB/best_threshold.txt",
                      repo_id=self.repo_id, repo_type=self.repo_type
                      )


    except Exception as ex:
      print(f"Exception: {ex}")
      traceback.print_exc()
    finally:
      print('-'*50)

