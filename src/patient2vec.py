from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import timeit
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

data_path = '/content/drive/My Drive/DL4H/DL4H_Project/Bi_dimensional/data'
model_path = '/content/drive/My Drive/DL4H/DL4H_Project/Bi_dimensional/models'
#data_path = '/content/drive/My Drive/DL4H/Project/Paper46/Data/MIMIC-III/Output'

"""# Reading Data"""

data= pd.read_pickle(data_path + '/Data-3_100.pkl')
data.rename(columns={'itemid_x': 'symptom', 'formulary_drug_cd': 'treatment', 'icd9_code': 'diagnosis'}, inplace=True)

data.head()

data.drop(columns = 'itemid_y', inplace= True)

data.shape

#data = data.iloc[0: 30000, :]

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

diag = data['diagnosis'].to_numpy()
diagnosis = set()
for s in diag:
  diagnosis.update(s)
vocab_diagnosis = list(diagnosis)
print(vocab_diagnosis)
print(len(vocab_diagnosis))

"""# Multi-hot Encoding"""

symptom_encoder = MultiLabelBinarizer()
treatment_encoder = MultiLabelBinarizer()
diagnosis_encoder = MultiLabelBinarizer()

mh_symptom = symptom_encoder.fit_transform(data['symptom'])

data['mh_symptom'] = symptom_encoder.fit_transform(data['symptom']).tolist()
data['mh_treatment'] = treatment_encoder.fit_transform(data['treatment']).tolist()
data['mh_diagnosis'] = diagnosis_encoder.fit_transform(data['diagnosis']).tolist()

print(data.columns)
print(data.head(5))

"""#Group by Patient"""

data_gb = data.groupby('subject_id')

'''for name, group in data_gb:
  print(group['symptom'], name)'''
  
subjects= data['subject_id'].unique()

"""#MIMIC-III Dataset

## Collate Function
"""

def collate_fn(batch):
  symptoms = []
  treatments = []
  diagnoses = []
  for patient in batch:
    symptoms.append(patient[0])
    treatments.append(patient[1])
    diagnoses.append(patient[2])

  symptoms = pad_sequence(symptoms, batch_first=True)
  treatments = pad_sequence(treatments, batch_first=True)
  diagnoses = pad_sequence(diagnoses, batch_first=True)

  return symptoms, treatments, diagnoses

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
        d = torch.from_numpy(np.array(group['mh_diagnosis'].values.tolist(),dtype = 'float32'))

        return [s, t, d]

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

    self.fc1 = nn.Linear(in_features = hidden_size, out_features = diagnosis_size)

  def forward(self, o_hidden):
      #return torch.rand(o_hidden.shape[0], o_hidden.shape[1], len(vocab_diagnosis), requires_grad=True)
      return torch.sigmoid(self.fc1(o_hidden))

"""#Model"""

device = check_device()
hidden_size = 16
pat2vec = Patient2Vec(len(vocab_symptom), len(vocab_treatment) ,hidden_size= hidden_size).to(device)
diag_pred = Diagnosis_Pred(hidden_size= hidden_size, diagnosis_size=len(vocab_diagnosis)).to(device)

params = list(pat2vec.parameters()) + list(diag_pred.parameters())

optimizer = torch.optim.Adam(params, lr=1e-3)
criterion= nn.BCELoss()

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

"""# Plotting"""

def my_plot(epochs, loss, ylabel):
    plt.plot(epochs, loss)
    plt.xlabel('Epoch No')
    plt.ylabel(ylabel)
    plt.show()

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

num_epochs = 20
loss_vals =  []
auc_all =[]
recall_all =[]
total_time = 0

pat2vec.train()
diag_pred.train()

for epoch in range(num_epochs):
    #print("Epoch ", epoch+1)
    start = timeit.default_timer()
    yhat_all = []
    true_all = []
    epoch_loss= []
    pred_all = []
    for i, (symptom, treatment, diagnosis) in enumerate(train_loader):
        #print(i)
        batch_size = symptom.shape[0]
        h = pat2vec.initHidden(batch_size).to(device)
        true = diagnosis.detach().numpy()

        optimizer.zero_grad()

        output = []
        visit_count = symptom.shape[1]
        symptom = symptom.to(device)
        treatment = treatment.to(device)
        for visit in range(visit_count):
            s = symptom[:, visit, :]
            t = treatment[:, visit, :]
            h = pat2vec(s, t, h)
            output.append(h)

        output = torch.stack(output, dim=1)
        yhat = diag_pred(output)

        pred = yhat.float().detach().cpu().numpy()
        yhat_np = yhat.float().detach().cpu().numpy()
        pred = np.where(pred > 0.5, 1, 0)
        
        true_all.extend(true)
        yhat_all.extend(yhat_np)
        pred_all.extend(pred)

        loss = criterion(yhat, diagnosis.to(device))
        loss.backward(retain_graph=True)
        epoch_loss.append(loss.item())
        optimizer.step()


    loss_vals.append(sum(epoch_loss)/len(epoch_loss))
    auc= auc_roc(yhat_all, true_all)
    auc_all.append(auc)
    rec = recall(pred_all, true_all)
    recall_all.append(rec)


    print("Loss:", loss_vals[epoch], "AUC:", auc,  "Recall", rec)

    stop = timeit.default_timer()
    total_time+= stop - start
    print('Time: ', stop - start) 
# plotting
my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals, 'BCE Loss')

my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), auc_all, 'AUC')
my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), recall_all, 'RECALL')
print(total_time/100)
#torch.save(pat2vec.state_dict(), model_path+"/e100_h32.pt")

"""#Validation"""

pat2vec.eval()
diag_pred.eval()
correct = np.zeros(len(vocab_diagnosis))

correct1 = 0
yhat_all = []
true_all = []
pred_all =[]
row_count = 0
for i, (symptom, treatment, diagnosis) in enumerate(val_loader):
        batch_size = symptom.shape[0]
        h = pat2vec.initHidden(batch_size).to(device)
        true = diagnosis.detach().numpy()

        # compute the model output
        output = []
        visit_count = symptom.shape[1]
        symptom = symptom.to(device)
        treatment = treatment.to(device)
        for visit in range(visit_count):
            s = symptom[:, visit, :]
            t = treatment[:, visit, :]
            o = pat2vec(s, t, h)
            output.append(o)
        #print(output)
        output = torch.stack(output, dim=1)
        yhat = diag_pred(output)
        #print(yhat)

        pred = yhat.float().detach().cpu().numpy()
        yhat_np = yhat.float().detach().cpu().numpy()
        pred = np.where(pred > 0.5, 1, 0)
        
        true_all.extend(true)
        #print(len(true_all))
        yhat_all.extend(yhat_np)
        pred_all.extend(pred)
        match = np.where(true ==  pred, 1, 0)
        #print(match.shape) # b * v * d

        correct1 += np.sum(match)
        
        match = np.sum(match, axis=1) # b * d
        #print(match.shape)
        match = np.sum(match, axis=0) # d
        #print(match.shape)
        
        correct += match

        row_count += true.shape[0] * true.shape[1] 

try:
    auc= auc_roc(yhat_all, true_all)
    auprc = auprc_(yhat_all, true_all)
    prec = precision(pred_all, true_all)
    rec =recall(pred_all, true_all)
except ValueError:
    pass


acc = correct/row_count
acc= np.mean(acc)

print("auc:", auc, "auprc: " , auprc, "acc: ", acc, "precision: ", prec, "recall: ", rec )

pat2vec.eval()
diag_pred.eval()
correct = np.zeros(len(vocab_diagnosis))

correct1 = 0
yhat_all = []
true_all = []
pred_all =[]
row_count = 0
for i, (symptom, treatment, diagnosis) in enumerate(test_loader):
        batch_size = symptom.shape[0]
        h = pat2vec.initHidden(batch_size).to(device)
        true = diagnosis.detach().numpy()

        # compute the model output
        output = []
        visit_count = symptom.shape[1]
        symptom = symptom.to(device)
        treatment = treatment.to(device)
        for visit in range(visit_count):
            s = symptom[:, visit, :]
            t = treatment[:, visit, :]
            o = pat2vec(s, t, h)
            output.append(o)
        #print(output)
        output = torch.stack(output, dim=1)
        yhat = diag_pred(output)
        #print(yhat)

        pred = yhat.float().detach().cpu().numpy()
        yhat_np = yhat.float().detach().cpu().numpy()
        pred = np.where(pred > 0.5, 1, 0)
        
        true_all.extend(true)
        #print(len(true_all))
        yhat_all.extend(yhat_np)
        pred_all.extend(pred)
        match = np.where(true ==  pred, 1, 0)
        #print(match.shape) # b * v * d

        correct1 += np.sum(match)
        
        match = np.sum(match, axis=1) # b * d
        #print(match.shape)
        match = np.sum(match, axis=0) # d
        #print(match.shape)
        
        correct += match

        row_count += true.shape[0] * true.shape[1] 

try:
    auc= auc_roc(yhat_all, true_all)
    auprc = auprc_(yhat_all, true_all)
    prec = precision(pred_all, true_all)
    rec =recall(pred_all, true_all)
except ValueError:
    pass


acc = correct/row_count
acc= np.mean(acc)

print("auc:", auc, "auprc: " , auprc, "acc: ", acc, "precision: ", prec, "recall: ", rec )
