import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

"""# Device"""

def check_device():
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
  else:
    device = torch.device("cpu")
    print("Running on the CPU")
  return device

"""# Data folder"""

#data_path = '/content/drive/My Drive/DL4H/DL4H_Project/Bi_dimensional/data'
#model_path = '/content/drive/My Drive/DL4H/DL4H_Project/Bi_dimensional/models'
data_path = '/content/drive/My Drive/DL4H/Project/Paper46/Data/MIMIC-III/Output'

"""# Reading Data"""

data_ohe = pd.read_pickle(data_path + '/OHE-100.pkl')

data_ohe = data_ohe.drop(columns=['itemid', 'formulary_drug_cd'])

data_ohe.head()

data_w2v= pd.read_pickle(data_path + '/W2V-100.pkl')

data_w2v.head()

#data.rename(columns={'itemid': 'symptom', 'formulary_drug_cd': 'treatment', 'icd9_code': 'diagnosis'}, inplace=True)

data = data_ohe.merge(data_w2v, on=['subject_id', 'hadm_id'], how = 'inner')

data.rename(columns={"icd9_code_x": "OHE_icd", "icd9_code_y": "W2V_icd"}, inplace = True)

data['icd9_size']=data['W2V_icd'].apply(lambda x: len(x))

data['icd9_size'].value_counts()

data=data[data['icd9_size']==10]

print(data)

data.rename(columns={'itemid': 'symptom', 'formulary_drug_cd': 'treatment'}, inplace=True)

"""# Symptom Vocabulary

"""

symp =  data['symptom'].to_numpy()
symptom = set()
for s in symp:
  symptom.update(s)
vocab_symptom = list(symptom)
print(vocab_symptom)
print(len(vocab_symptom))

"""# Treatment Vocabulary

"""

treat =  data['treatment'].to_numpy()
treatment = set()
for s in treat:
  treatment.update(s)
vocab_treatment = list(treatment)
print(vocab_treatment)
print(len(vocab_treatment))

"""# Diagnosis Vocabulary"""

'''diag = data['diagnosis'].to_numpy()
diagnosis = set()
for s in diag:
diagnosis.update(s)
vocab_diagnosis = list(diagnosis)
print(vocab_diagnosis)
print(len(vocab_diagnosis))'''

"""Multi"""

symptom_encoder = MultiLabelBinarizer()
treatment_encoder = MultiLabelBinarizer()
#diagnosis_encoder = MultiLabelBinarizer()

mh_symptom = symptom_encoder.fit_transform(data['symptom'])

data['mh_symptom'] = symptom_encoder.fit_transform(data['symptom']).tolist()
data['mh_treatment'] = treatment_encoder.fit_transform(data['treatment']).tolist()
#data['mh_diagnosis'] = diagnosis_encoder.fit_transform(data['diagnosis']).tolist()

print(data.columns)
print(data.head(5))

"""#Group by Patient"""

data_gb = data.groupby('subject_id')

'''for name, group in data_gb:
  print(group['symptom'], name)'''
  
subjects= data['subject_id'].unique()

print(data_gb.head())

"""#MIMIC-III Dataset

## Collate Function
"""

def collate_fn(batch):
  symptoms = []
  treatments = []
  diagnoses_ohe = []
  diagnoses_w2v = []
  for patient in batch:
    symptoms.append(patient[0])
    treatments.append(patient[1])
    diagnoses_ohe.append(patient[2])
    diagnoses_w2v.append(patient[3])
    #print(patient[2])

  symptoms = pad_sequence(symptoms, batch_first=True)
  treatments = pad_sequence(treatments, batch_first=True)
  diagnoses_ohe = pad_sequence(diagnoses_ohe, batch_first=True)
  diagnoses_w2v = pad_sequence(diagnoses_w2v, batch_first=True)

  return symptoms, treatments, diagnoses_ohe, diagnoses_w2v

"""## Custom Dataset"""

class Mimic3(Dataset):
    # load the dataset
    def __init__(self,data_gb, subjects):
        self.subjects= subjects
        self.data_gb = data_gb

    # number of rows in the dataset
    def __len__(self):
        return self.subjects.shape[0]

    # get a row at an index
    def __getitem__(self, idx):
        subject_id=subjects[idx]
        group = self.data_gb.get_group(subject_id)
        s = torch.from_numpy(np.array(group['mh_symptom'].values.tolist(), dtype = 'float32') )
        t = torch.from_numpy(np.array(group['mh_treatment'].values.tolist(), dtype = 'float32') )
        d_ohe = torch.from_numpy(np.array(group['OHE_icd'].values.tolist(),dtype = 'float32'))
        d_w2v = torch.from_numpy(np.array(group['W2V_icd'].values.tolist(),dtype = 'float32'))

        print(s.shape, t.shape,d_ohe.shape, d_w2v.shape)
        return [s, t, d_ohe, d_w2v]

class Diag(Dataset):
    def __init__(self,data):
        self.w2v = data['W2V_icd'].to_numpy()
        self.ohe = data['OHE_icd'].to_numpy()

    def __len__(self):
        return self.w2v.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.w2v[idx]), torch.FloatTensor(self.ohe[idx])

"""#Patient2Vec"""

class Patient2Vec(nn.Module):
  def __init__(self, symptom_size, treatment_size, hidden_size=16):
    super(Patient2Vec, self).__init__()
    self.hidden_size = hidden_size
    self.s2h = nn.Linear(symptom_size + hidden_size, 3*hidden_size, bias= True)
    self.t2h = nn.Linear(treatment_size + hidden_size, 3*hidden_size, bias = True)
    self.h2o = nn.Linear(3*hidden_size, 2*hidden_size, bias= False)
    self.o2o = nn.Linear(2*hidden_size,hidden_size, bias=True)

  def forward(self, symptom, treatment, hidden):
      combined_symptom = torch.cat((symptom, hidden), 1)
      combined_treatment = torch.cat((treatment, hidden), 1)
      
      a_s = self.s2h(combined_symptom)
      a_t = self.t2h(combined_treatment)
      hadam = a_s*a_t
      ho = torch.tanh(self.h2o(hadam))
      oo = self.o2o(ho)
      return oo

  def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

"""#Diagnosis Prediction"""

class Diagnosis_Pred(nn.Module):
  def __init__(self, hidden_size, diagnosis_size):
    super(Diagnosis_Pred, self).__init__()

    self.fc1 = nn.ModuleList([nn.Linear(in_features = hidden_size, out_features = diagnosis_size) for i in range(10)])

  def forward(self, o_hidden):
      outputs = []
      for fc in self.fc1:
            outputs.append(fc(o_hidden))
      output = torch.stack(outputs, dim=2)
      return output

"""#Word2vec to vocab length"""

class W2V2OHE(nn.Module):
  def __init__(self, hidden_size, diagnosis_size):
    super(W2V2OHE, self).__init__()
    self.fc1 = nn.Linear(in_features=hidden_size, out_features=diagnosis_size)

  def forward(self, x):
    return torch.softmax(self.fc1(x), dim=0)

"""# Plotting"""

def my_plot(epochs, loss, ylabel):
    plt.plot(epochs, loss)
    plt.xlabel('Epoch No')
    plt.ylabel(ylabel)
    plt.show()

"""#Model"""

device = check_device()

w2v2ohe = W2V2OHE(hidden_size = 200, diagnosis_size = 1288).to(device)
params = list(w2v2ohe.parameters())
criterion1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(params, lr=1e-3)

full_set1 = Diag(data)
total_size1 = len(full_set1)
val_size1 = int(0.1 * total_size1)
test_size1 = val_size1
train_size1 = total_size1 - val_size1 - test_size1
train_set1, val_set1, test_set1 = torch.utils.data.random_split(full_set1, [train_size1, val_size1, test_size1])

batch_size1 = 32
train_loader1 = DataLoader(train_set1, batch_size = batch_size1, shuffle = True)
val_loader1 = DataLoader(val_set1, batch_size = batch_size1, shuffle = False)
test_loader1 = DataLoader(test_set1, batch_size = batch_size1, shuffle = False)

num_epochs1 = 25
loss_vals1 =  []

w2v2ohe.train()

for epoch in range(num_epochs1):
    print("Epoch ", epoch+1)
    epoch_loss = []
    for i, (w2v, ohe) in enumerate(train_loader1):
        w2v = w2v.to(device)
        ohe = ohe.to(device)

        pred = w2v2ohe(w2v.float())

        loss = criterion1(pred, ohe)
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer1.step()
    loss_vals1.append(np.mean(epoch_loss))
    print(loss_vals1[epoch])

my_plot(np.linspace(1, num_epochs1, num_epochs1).astype(int), loss_vals1, 'CrossEntropyLoss')

torch.save(w2v2ohe.state_dict(), data_path + '/w2v2ohe.pt')

vocab_diagnosis_len = 1288

pat2vec = Patient2Vec(len(vocab_symptom), len(vocab_treatment) ,hidden_size= 8).to(device)
diag_pred =  Diagnosis_Pred(hidden_size= 8, diagnosis_size=200).to(device)

params = list(pat2vec.parameters()) + list(diag_pred.parameters())

optimizer = torch.optim.Adam(params, lr=1e-3)
criterion= nn.MSELoss(delta=0.01 )

"""#Train, Validation, Test Dataset"""

full_set = Mimic3(data_gb, subjects)
total_size = len(full_set)
val_size = int(0.1 * total_size)
test_size = val_size
train_size = total_size - val_size - test_size
train_set, val_set, test_set = torch.utils.data.random_split(full_set, [train_size, val_size, test_size])

"""# Dataloader"""

batch_size =8
train_loader= DataLoader(train_set, batch_size= batch_size, shuffle = True, collate_fn=collate_fn)
val_loader= DataLoader(val_set, batch_size= batch_size, shuffle = False, collate_fn=collate_fn)
test_loader= DataLoader(test_set, batch_size= batch_size, shuffle = False, collate_fn=collate_fn)

"""# Parameters"""

pat2vec_params = sum(p.numel() for p in pat2vec.parameters())
pat2vec_train_params = sum(p.numel() for p in pat2vec.parameters() if p.requires_grad)
print(pat2vec_params, pat2vec_train_params)

"""# Evaluation

## Accuracy
"""

def calc_accuracy(true, pred):
  print(true.shape)
  print(pred.shape)

  print(true[0].shape, true[1].shape)
  print(pred[0].shape, pred[1].shape)
  pred_ = np.where(pred > 0.5, 1, 0)
  acc = np.where(pred == true, 1, 0)
  acc = np.mean(acc)
  print('Accuracy: ', acc)
  return acc

"""## Precision"""

def auc_roc(pred, true):
  #pred = np.array(pred[:, :, 0]).reshape(-1)
  #true = np.array(true[:, :, 0]).reshape(-1)


  all_true= []
  all_pred = []

  for i in pred:
    for j in i:
      for k in j:
        all_pred.append(k)

  for i in true:
    for j in i:
      for k in j:
        all_true.append(k)
  
  return roc_auc_score(all_true, all_pred)

  #print(len(all_true))

def auprc_(pred, true):
  all_true= []
  all_pred = []

  for i in pred:
    for j in i:
      for k in j:
        all_pred.append(k)

  for i in true:
    for j in i:
      for k in j:
        all_true.append(k)

  precision, recall, thresholds = precision_recall_curve(all_true, all_pred)
  return metrics.auc(recall, precision)

def precision(pred, true):
  all_true= []
  all_pred = []

  for i in pred:
    for j in i:
      for k in j:
        all_pred.append(k)

  for i in true:
    for j in i:
      for k in j:
        all_true.append(k)

  precision= precision_score(all_true, all_pred, average='micro')
  return precision

def recall(pred, true):
  all_true= []
  all_pred = []

  for i in pred:
    for j in i:
      for k in j:
        all_pred.append(k)

  for i in true:
    for j in i:
      for k in j:
        all_true.append(k)

  recall= recall_score(all_true, all_pred, average='micro')
  return recall

"""# Training"""

num_epochs = 10
loss_vals =  []

pat2vec.train()
diag_pred.train()
for epoch in range(num_epochs):
    print("Epoch ", epoch+1)
    yhat_all = []
    true_all = []
    row_count = 0
    epoch_loss= []
    pred_all = []
    for i, (symptom, treatment, diagnosis_ohe, diagnosis_w2v) in enumerate(train_loader):
        batch_size = symptom.shape[0]
        h = pat2vec.initHidden(batch_size).to(device)
        true = diagnosis_w2v.detach().numpy()
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        output = []
        visit_count = symptom.shape[1]
        print(visit_count)
        symptom = symptom.to(device)
        treatment = treatment.to(device)
        for visit in range(visit_count):
            s = symptom[:, visit, :]
            t = treatment[:, visit, :]
            h = pat2vec(s, t, h)
            output.append(h)
        #print(output)
        output = torch.stack(output, dim=1)
        yhat = diag_pred(output)
        print(yhat.shape, diagnosis_w2v.shape)
        
        loss = criterion1(yhat, diagnosis_w2v.to(device))
        # credit assignment
        loss.backward(retain_graph=True)
        epoch_loss.append(loss.item())
        # update model weights
        optimizer.step()
        #print(len(yhat_all))

    #print(len(yhat_all))
    loss_vals.append(sum(epoch_loss)/len(epoch_loss))
    

    print("Loss:", loss_vals[epoch])
# plotting
my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals, 'MSE Loss')

#torch.save(pat2vec.state_dict(), model_path+"/e100_h32.pt")

torch.save(pat2vec.state_dict(), data_path + '/pat2vec.pt')
torch.save(diag_pred.state_dict(), data_path + '/diag_pred.pt')

from torch.nn.utils.rnn import pad_sequence
a = torch.ones(5, 4, 3)
b = torch.ones(2, 4, 3)
c = torch.ones(3, 4, 3)
p = pad_sequence([a, b, c], batch_first=True)
print(p)

"""#Validation"""

pat2vec = Patient2Vec(len(vocab_symptom), len(vocab_treatment), hidden_size= 8).to(device)
pat2vec.load_state_dict(torch.load(data_path + '/pat2vec.pt'))
pat2vec.eval()

diag_pred = Diagnosis_Pred(hidden_size= 8, diagnosis_size=200).to(device)
diag_pred.load_state_dict(torch.load(data_path + '/diag_pred.pt'))
diag_pred.eval()

w2v2ohe = W2V2OHE(hidden_size = 200, diagnosis_size = 1288).to(device)
w2v2ohe.load_state_dict(torch.load(data_path + '/w2v2ohe.pt'))
w2v2ohe.eval()

yhat_all = []
true_all = []
pred_all =[]

for i, (symptom, treatment, diagnosis_ohe, diagnosis_w2v) in enumerate(test_loader):
        batch_size = symptom.shape[0]
        h = pat2vec.initHidden(batch_size).to(device)
        true = diagnosis_ohe.detach().numpy()
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        output = []
        visit_count = symptom.shape[1]
        print(visit_count)
        symptom = symptom.to(device)
        treatment = treatment.to(device)
        for visit in range(visit_count):
            s = symptom[:, visit, :]
            t = treatment[:, visit, :]
            h = pat2vec(s, t, h)
            output.append(h)
        #print(output)
        output = torch.stack(output, dim=1)
        yhat = diag_pred(output)
        yhat = w2v2ohe(yhat)
        print(yhat.shape, diagnosis_ohe.shape)

        shape = yhat.shape

        yhat = yhat.reshape(shape[0], shape[1], -1)
        diagnosis_ohe = diagnosis_ohe.reshape(shape[0], shape[1], -1)
        
        print(yhat.shape, diagnosis_ohe.shape)

        pred = yhat.float().detach().cpu().numpy()
        yhat_np = yhat.float().detach().cpu().numpy()
        pred = np.where(pred > 0.5, 1, 0)
        
        true_all.extend(diagnosis_ohe)
        #print(len(true_all))
        yhat_all.extend(yhat_np)
        pred_all.extend(pred)

try:
    auc= auc_roc(yhat_all, true_all)
    rec =recall(pred_all, true_all)
except ValueError:
    print('Failed')
    pass

print("auc:", auc, "recall: ", rec )

print(yhat_all[0].shape)
print(true_all[0].shape)
