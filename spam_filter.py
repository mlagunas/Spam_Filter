# Manuel Lagunas Arto
# 4th February 2016

######################################################
# Imports
######################################################

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import json
import glob
from sklearn import metrics
import random
from sklearn.cross_validation import KFold

######################################################
# Aux. functions
######################################################

# load_enron_folder: load training, validation and test sets from an enron path
def load_enron_folder(path):

   ### Load ham mails ###

   # List mails in folder
   ham_folder = path + '\ham\*.txt'
   ham_list = glob.glob(ham_folder)
   num_ham_mails = len(ham_list)

   ham_mail = []
   for i in range(0,num_ham_mails):
      ham_i_path = ham_list[i]
      # Open file
      ham_i_file = open(ham_i_path, 'r')
      # Read
      ham_i_str = ham_i_file.read()
      # Convert to Unicode
      ham_i_text = ham_i_str.decode('utf-8',errors='ignore')
      # Append to the mail structure
      ham_mail.append(ham_i_text)
      # Close file
      ham_i_file.close()

      random.shuffle(ham_mail)

   # Load spam mails

   spam_folder = path + '\spam\*.txt'
   spam_list = glob.glob(spam_folder)
   num_spam_mails = len(spam_list)

   spam_mail = []
   for i in range(0,num_spam_mails):
      spam_i_path = spam_list[i]
      # Open file
      spam_i_file = open(spam_i_path, 'r')
      # Read
      spam_i_str = spam_i_file.read()
      # Convert to Unicode
      spam_i_text = spam_i_str.decode('utf-8',errors='ignore')
      # Append to the mail structure
      spam_mail.append(spam_i_text)
      # Close file
      spam_i_file.close()

      random.shuffle(spam_mail)

   # Separate into training, validation and test
   num_ham_training = int(round(0.8*num_ham_mails))
   ham_training_mail = ham_mail[0:num_ham_training]
   #print(num_ham_mails)
   #print(num_ham_training)
   #print(len(ham_training_mail))
   ham_training_labels = [0]*num_ham_training
   #print(len(ham_training_labels))

   num_ham_validation = int(round(0.1*num_ham_mails))
   ham_validation_mail = ham_mail[num_ham_training:num_ham_training+num_ham_validation]
   #print(num_ham_validation)
   #print(len(ham_validation_mail))
   ham_validation_labels = [0] * num_ham_validation
   #print(len(ham_validation_labels))

   ham_test_mail = ham_mail[num_ham_training+num_ham_validation:num_ham_mails]
   #print(num_ham_mails-num_ham_training-num_ham_validation)
   #print(len(ham_test_mail))
   ham_test_labels = [0] * (num_ham_mails-num_ham_training-num_ham_validation)
   #print(len(ham_test_labels))

   num_spam_training = int(round(0.8*num_spam_mails))
   spam_training_mail = spam_mail[0:num_spam_training]
   #print(num_spam_mails)
   #print(num_spam_training)
   #print(len(spam_training_mail))
   spam_training_labels = [1]*num_spam_training
   #print(len(spam_training_labels))

   num_spam_validation = int(round(0.1*num_spam_mails))
   spam_validation_mail = spam_mail[num_spam_training:num_spam_training+num_spam_validation]
   #print(num_spam_validation)
   #print(len(spam_validation_mail))
   spam_validation_labels = [1] * num_spam_validation
   #print(len(spam_validation_labels))

   spam_test_mail = spam_mail[num_spam_training+num_spam_validation:num_spam_mails]
   #print(num_spam_mails-num_spam_training-num_spam_validation)
   #print(len(spam_test_mail))
   spam_test_labels = [1] * (num_spam_mails-num_spam_training-num_spam_validation)
   #print(len(spam_test_labels))

   training_mails = ham_training_mail + spam_training_mail
   training_labels = ham_training_labels + spam_training_labels
   validation_mails = ham_validation_mail + spam_validation_mail
   validation_labels = ham_validation_labels + spam_validation_labels
   test_mails = ham_test_mail + spam_test_mail
   test_labels = ham_test_labels + spam_test_labels

   data = {'training_mails': training_mails, 'training_labels': training_labels, 'validation_mails': validation_mails, 'validation_labels': validation_labels, 'test_mails': test_mails, 'test_labels': test_labels} 

   return data

def kfold_crossvalidation(learner,k,n,examples,cv,labels):
   bestPip = 0
   bestErrV = 999999
   bestErrT = 999999
   bestAlph = 999999
   if(k<2):
      Error("Error n in Kfold must be >= 2")
   for size in range (1,n+1):
      #Factory para la creacion del pipeline con la distribucion que modelara nuestra red
      #asi como la bolsa de palabras correspondiente en cada caso
      if (learner=="MultinomialNB CV"):
         pipeline = Pipeline([\
            ('vect', cv),\
            ('clf', MultinomialNB(size ,fit_prior=True, class_prior=None)),])
      if (learner=="MultinomialNB N"):
         pipeline = Pipeline([\
            ('vect', cv),\
            ('tfidf', TfidfTransformer()),\
            ('clf', MultinomialNB(size ,fit_prior=True, class_prior=None)),])
      if (learner=="BernoulliNB"):
         pipeline = Pipeline([\
            ('vect', cv),\
            ('clf', BernoulliNB(size, fit_prior=True, class_prior=None)),])
      
      i = 1
      errV = 0
      errT = 0
      kf = KFold(len(examples),k)

      for train_index,validation_index in kf:
         #Extraemos las palabras necesarias
         if i == 1 or i == k:
            train = examples[train_index[0]:train_index[-1]]
            train_labels = labels[train_index[0]:train_index[-1]]
            validation = examples[validation_index[0]:validation_index[-1]]
            validation_labels = labels[validation_index[0]:validation_index[-1]]
         else:
            train = examples[train_index[0]:validation_index[0]-1] + examples[validation_index[-1]+1:train_index[-1]]
            train_labels = labels[train_index[0]:validation_index[0]-1] + labels[validation_index[-1]+1:train_index[-1]]
            validation = examples[validation_index[0]:validation_index[-1]]
            validation_labels = labels[validation_index[0]:validation_index[-1]]
         i += 1
         
         #Entrenamos la red y calculamos los errores
         pipeline.fit(train,train_labels)
         predicted = pipeline.predict(validation)
         errV += metrics.f1_score(validation_labels, predicted)
         predicted = pipeline.predict(train)
         errT += metrics.f1_score(train_labels, predicted)
      #Hacemos el calculo medio del error
      errT = errT/k
      errV = errV/k
      if(bestErrV > errV):
         bestErrV = errV
         bestErrT = errT
         bestPip = pipeline
         bestAlph = size
   return [bestN,errV,errT,bestAlph]

def load(i, folder):
   data = load_enron_folder(folder + `i`)
   print "loaded enron", i
   return data

######################################################
# Main
######################################################
PATH = r'PATH TO ENRON FOLDERS'

print("Starting...")

# Path to the folder containing the mails

data1 = load(1,PATH)
data2 = load(2,PATH)
data3 = load(3,PATH)
data4 = load(4,PATH)
data5 = load(5,PATH)
data6 = load(6,PATH)

#Set variables
training_mails = data1['training_mails']+data2['training_mails']+data3['training_mails']+data4['training_mails']+data5['training_mails']+data6['training_mails']
training_labels = data1['training_labels']+data2['training_labels']+data3['training_labels']+data4['training_labels']+data5['training_labels']+data6['training_labels']
validation_mails = data1['validation_mails']+data2['validation_mails']+data3['validation_mails']+data4['validation_mails']+data5['validation_mails']+data6['validation_mails']
validation_labels = data1['validation_labels']+data2['validation_labels']+data3['validation_labels']+data4['validation_labels']+data5['validation_labels']+data6['validation_labels']
test_mails = data1['test_mails']+data2['test_mails']+data3['test_mails']+data4['test_mails']+data5['test_mails']+data6['test_mails']
test_labels = data1['test_labels']+data2['test_labels']+data3['test_labels']+data4['test_labels']+data5['test_labels']+data6['test_labels']

training = training_mails+validation_mails
labels = training_labels+validation_labels

ALPHA = 1
bestErr = [9999]

def get_falses(prediction, label, mail):
   false_ham = []
   false_spam = []
   for i in range (0,len(prediction)):
      if prediction[i] == 0 and label[i] == 1:
         false_spam.append(mail[i])
      elif prediction[i] == 1 and label[i] == 0:
         false_ham.append(mail[i])
   return [false_ham,false_spam]

def print_result(text,model, norm = False, dist=MultinomialNB(ALPHA ,fit_prior=True, class_prior=None)):
   global bestErr
   if(norm):
      pipeline = Pipeline([\
                  ('vect',model),\
                  ('tfidf', TfidfTransformer()),\
                  ('clf', dist),])
   else:
      pipeline = Pipeline([\
                  ('vect',model),\
                  ('clf', dist),])
   pipeline.fit(training,labels)
   predicted = pipeline.predict(test_mails)
   predictedT = pipeline.predict(training)
   errV = 1 - metrics.f1_score(test_labels, predicted)
   errT = 1 - metrics.f1_score(labels, predictedT)
   if(errV < bestErr[0]):
      bestErr = [errV,text,predicted]
   print text,"================"\
      "\n\terrV:",errV, \
      "\n\terrT:",errT

print_result("MultinomialNB palabras",CountVectorizer(binary = False))
print_result("MultinomialNB mezcla(palabra/bigrama)",CountVectorizer(ngram_range=(1,2), binary = False))
print_result("MultinomialNB bigramas",CountVectorizer(ngram_range=(2,2), binary = False))

print_result("MultinomialNB normalizada palabras",CountVectorizer(binary = False), True)
print_result("MultinomialNB normalizada mezcla(palabra/bigrama)",CountVectorizer(ngram_range=(1,2), binary = False), True)
print_result("MultinomialNB normalizada bigramas",CountVectorizer(ngram_range=(2,2), binary = False), True)

print_result("Multinomial binarizada palabras",CountVectorizer(ngram_range=(1,1), binary = True))
print_result("Multinomial binarizada mezcla(palabra/bigrama)",CountVectorizer(ngram_range=(1,2), binary = True))
print_result("Multinomial binarizada bigramas",CountVectorizer(ngram_range=(2,2), binary = True))

print_result("Bernouilli palabras",CountVectorizer(binary = True),BernoulliNB(ALPHA, fit_prior=True, class_prior=None))
print_result("Bernouilli mezcla(palabra/bigrama)",CountVectorizer(ngram_range=(1,2), binary = True),BernoulliNB(ALPHA, fit_prior=True, class_prior=None))
print_result("Bernouilli bigramas",CountVectorizer(ngram_range=(2,2), binary = True),BernoulliNB(ALPHA, fit_prior=True, class_prior=None))

print "Mejor distribucion: ",bestErr[1], "con error de validacion = ", bestErr[0]
print "Matriz de confusion:", metrics.confusion_matrix(test_labels, bestErr[2])
precision, recall, thresholds = metrics.precision_recall_curve(test_labels, bestErr[2])
plt.clf()
plt.plot(precision, recall,label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
falses = get_falses(bestErr[2],test_labels,test_mails)
#raw_input("Pulsa enter para mostrar los FALSOS POSITIVOS")
print "Falso ham =======================\n",falses[0][0]
#raw_input("Pulsa enter para mostrar los FALSOS NEGATIVOS")
print "Falso spam =======================\n",falses[1][0]