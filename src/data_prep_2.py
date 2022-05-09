import pandas as pd
from datetime import datetime
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')

input_dir = '/content/drive/My Drive/DL4H/Project/Paper46/Data/MIMIC-III/Input'
output_dir = '/content/drive/My Drive/DL4H/Project/Paper46/Data/MIMIC-III/Output'

pd.set_option('display.max_rows', None)

"""# Treatment"""

prescriptions_cols = ['SUBJECT_ID', 'HADM_ID', 'FORMULARY_DRUG_CD']
prescriptions = pd.read_csv(input_dir + '/PRESCRIPTIONS.csv', sep=',', usecols=prescriptions_cols)

# Keep drugs that have been prescribed to at least 20 patients
prescriptions_gb = prescriptions[['SUBJECT_ID', 'FORMULARY_DRUG_CD']].drop_duplicates().groupby('FORMULARY_DRUG_CD')
prescriptions_gb = prescriptions_gb.size().reset_index(name='COUNT')
prescriptions_gb = prescriptions_gb[prescriptions_gb['COUNT'] >= 20]

prescriptions = prescriptions[prescriptions['FORMULARY_DRUG_CD'].isin(prescriptions_gb.FORMULARY_DRUG_CD)]
print(prescriptions.shape)
# to do -> filtering based on entire ehr frequency 
prescriptions = prescriptions.groupby(['SUBJECT_ID', 'HADM_ID'])['FORMULARY_DRUG_CD'].apply(list).to_frame().reset_index()

print(prescriptions.shape)
print(prescriptions.head)

"""# Admission"""

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
admissions_cols = ['SUBJECT_ID', 'HADM_ID', 'DISCHTIME']
admissions = pd.read_csv(input_dir + '/ADMISSIONS.csv', sep=',', usecols=admissions_cols, parse_dates=['DISCHTIME'], date_parser=dateparse)

print(admissions.shape)
print(admissions)

"""# Diagnosis"""

diagnoses_icd_cols = ['SUBJECT_ID',	'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']
diagnoses_icd = pd.read_csv(input_dir + '/DIAGNOSES_ICD.csv', sep=',', usecols=diagnoses_icd_cols)
diagnoses_icd = diagnoses_icd.dropna()

diagnoses_icd = pd.merge(diagnoses_icd, admissions, how="inner", on=['SUBJECT_ID', 'HADM_ID'])
diagnoses_icd = diagnoses_icd.sort_values(by=['SUBJECT_ID',	'HADM_ID', 'DISCHTIME'])

diagnoses_icd_gb = diagnoses_icd.groupby('ICD9_CODE').size().reset_index(name='COUNT')
diagnoses_icd_gb = diagnoses_icd_gb[diagnoses_icd_gb.COUNT > 50]
diagnoses_icd = diagnoses_icd[diagnoses_icd['ICD9_CODE'].isin(diagnoses_icd_gb.ICD9_CODE)]

diagnoses_icd_gb = diagnoses_icd.groupby(['SUBJECT_ID',	'HADM_ID']).size().reset_index(name='COUNT')
diagnoses_icd_gb = diagnoses_icd_gb[diagnoses_icd_gb.COUNT > 9]
diagnoses_icd = diagnoses_icd[diagnoses_icd['HADM_ID'].isin(diagnoses_icd_gb.HADM_ID)]

diagnoses_icd = diagnoses_icd.groupby(['SUBJECT_ID', 'HADM_ID']).apply(lambda x: x.nsmallest(n=10, columns='SEQ_NUM'))
diagnoses_icd = diagnoses_icd.drop(columns=['SUBJECT_ID', 'HADM_ID'])
diagnoses_icd = diagnoses_icd.reset_index()

"""## Word2Vec"""

def cleanData(sentence):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

    # remove stop words
    sentence = " ".join([word for word in sentence.split()
                        if word not in stopwords.words('english')])

    return sentence

def get_vector(model, sentence):
        # convert to lowercase, ignore all special characters - keep only
        # alpha-numericals and spaces
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

        vectors = [model.wv[w] for w in word_tokenize(sentence)
                   if w in model.wv]

        v = np.zeros(model.vector_size)

        if (len(vectors) > 0):
            v = (np.array([sum(x) for x in zip(*vectors)])) / v.size

        return v

d_icd_diagnosis_cols = ['ICD9_CODE', 'LONG_TITLE']
d_icd_diagnosis = pd.read_csv(input_dir + '/D_ICD_DIAGNOSES.csv', sep=',', usecols=d_icd_diagnosis_cols)
d_icd_diagnosis = d_icd_diagnosis.drop_duplicates(subset='LONG_TITLE')
d_icd_diagnosis['LONG_TITLE'] = d_icd_diagnosis['LONG_TITLE'].map(lambda x: cleanData(x))
titles = d_icd_diagnosis['LONG_TITLE'].values.tolist()
tok_titles = [word_tokenize(title) for title in titles]
model = Word2Vec(tok_titles, sg=1, size=200, window=10, min_count=5, workers=4, iter=100)
d_icd_diagnosis['W2V_TITLE'] = d_icd_diagnosis['LONG_TITLE'].map(lambda x: get_vector(model, x))
print(d_icd_diagnosis.head(20))
d_icd_diagnosis = d_icd_diagnosis.drop(columns=['LONG_TITLE'])

diagnoses_w2v = pd.merge(diagnoses_icd, d_icd_diagnosis, how='inner', on=['ICD9_CODE'])

print(diagnoses_w2v.head(20))

diagnoses_w2v = diagnoses_w2v.sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], ascending=True)
diagnoses_w2v = diagnoses_w2v.groupby(['SUBJECT_ID', 'HADM_ID'])['W2V_TITLE'].apply(list).to_frame().reset_index()
diagnoses_w2v = diagnoses_w2v.rename(columns={'W2V_TITLE': 'ICD9_CODE'})
print(diagnoses_w2v.head())

"""## One Hot Encoding"""

lb = LabelBinarizer()

diagnoses_icd['ICD9_OHE'] = lb.fit_transform(diagnoses_icd['ICD9_CODE']).tolist()
diagnoses_icd = diagnoses_icd.sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], ascending=True)
diagnoses_icd = diagnoses_icd.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_OHE'].apply(list).to_frame().reset_index()
diagnoses_icd = diagnoses_icd.rename(columns={'ICD9_OHE': 'ICD9_CODE'})
print(diagnoses_icd.head())

print(diagnoses_icd.shape)
print(diagnoses_icd.head())

"""# Treatment & Diagnosis"""

df = pd.merge(diagnoses_icd, prescriptions, how='inner', on=['SUBJECT_ID', 'HADM_ID'])
print(df.shape)
print(df.head())

"""# Symptom"""

def merge_item_list(item_list):
  all_items = []
  total_len = 0
  for index, value in item_list.items():
      total_len += len(value)
      all_items.extend(value)
  if(total_len != len(all_items)):
    print("ERROR!! Lengths mismatch.")
  return all_items

chunksize = 10**6
chunk_list = []
chart_events_cols = ['SUBJECT_ID', 'HADM_ID', 'ITEMID']
chunk_reader = pd.read_csv(input_dir + '/CHARTEVENTS.csv', sep=',', usecols=chart_events_cols, chunksize=chunksize)
chunk_no = 1
for chunk in chunk_reader:
  print('chunk_no', chunk_no)
  new_chunk = chunk[chunk.set_index(['SUBJECT_ID', 'HADM_ID']).index.isin(df.set_index(['SUBJECT_ID', 'HADM_ID']).index)]
  print(new_chunk.shape)
  new_chunk = new_chunk.groupby(['SUBJECT_ID', 'HADM_ID'])['ITEMID'].apply(list).to_frame().reset_index()
  print(new_chunk.shape)
  chunk_list.append(new_chunk)
  chunk_no += 1
  if(chunk_no > 100):
    break

chart_events = pd.concat(chunk_list, axis=0)
chart_events_gb = chart_events.groupby(['SUBJECT_ID', 'HADM_ID'])
chart_events = chart_events_gb.apply(lambda x: merge_item_list(x['ITEMID'])).to_frame().reset_index()
chart_events = chart_events.rename(columns={0: 'ITEMID'})
chart_events.to_pickle(output_dir + '/CHARTEVENTS.pkl')

print(chart_events.shape)
print(chart_events.head())

df = pd.merge(chart_events, df, how='inner', on=['SUBJECT_ID', 'HADM_ID'])
print(df.shape)
print(df.head())

df = df.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id', 'ITEMID': 'itemid', 'ICD9_CODE': 'icd9_code', 'FORMULARY_DRUG_CD': 'formulary_drug_cd'})
df.to_pickle(output_dir + '/OHE-100.pkl')
