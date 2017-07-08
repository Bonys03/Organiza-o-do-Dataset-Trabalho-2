import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict

train_dados = pd.read_csv("train.csv")
print(train_dados.head()) #IMPRESSÃO DE ALGUMAS LINHAS DO DATASET
print(train_dados.shape)  #IMPRESSÃO DA QUANTIDADE DE LINHAS E COLUNAS DO DATASET
print("")
alvo = train_dados["Category"].unique()
print(alvo) #IMPRESSÃO DE TODOS OS TIPOS DE CATEGORIAS NO DATASET
print("")
test_dados = pd.read_csv("test.csv")
print(test_dados.head()) #IMPRESSÃO DO ARQUIVO TEST
print(test_dados.shape) #NOTA: NÃO HÁ A COLUNA CATEGORY NEM A DE RESOLUTION
print("")
dados_dicionario = {}
count = 1
for dados in alvo:
    dados_dicionario[dados] = count
    count+=1
train_dados["Category"] = train_dados["Category"].replace(dados_dicionario) #SUBSTITUINDO AS CATEGORIAS POR VALORES NUMÉRICOS

#SUBSTITUINDO OS DIAS DA SEMANA POR NÚMEROS
dados_week_dicionario = {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}
train_dados["DayOfWeek"] = train_dados["DayOfWeek"].replace(dados_week_dicionario)
test_dados["DayOfWeek"] = test_dados["DayOfWeek"].replace(dados_week_dicionario)
#SUBSTITUINDO OS NOMES DOS DISTRITOS POR VALORES NUMÉRICOS
distrito = train_dados["PdDistrict"].unique()
dados_dicionario_distrito = {}
count = 1
for dados in distrito:
    dados_dicionario_distrito[dados] = count
    count+=1 
train_dados["PdDistrict"] = train_dados["PdDistrict"].replace(dados_dicionario_distrito)
test_dados["PdDistrict"] = test_dados["PdDistrict"].replace(dados_dicionario_distrito)

print(train_dados.head()) #IMPRESSÃO DOS DADOS MODIFICADOS DO DATASET

colunas_train = train_dados.columns
print(colunas_train) #IMPRESSÃO DE CADA COLUNA DO DATASET
colunas_test = test_dados.columns
print(colunas_test)	#IMPRESSÃO DE CADA COLUNA DO TEST

colunas = colunas_train.drop("Resolution")
print(colunas) #IMPRESSÃO SEM A COLUNA RESOLUTION

#NOTA: A COLUNA RESOLUTION FOI REMOVIDA COM O PROPÓSITO DE VERIFICAR A CORRELAÇÃO ENTRE CATEGORY E AS OUTRAS COLUNAS
#COMO RESOLUTION NÃO APARECE NO ARQUIVO TEST, ELA SERÁ REMOVIDA.

train_dados_new = train_dados[colunas]
print(train_dados_new.head()) #IMPRESSÃO DOS DADOS SEM A RESOLUTION

print(train_dados_new.describe()) #VERIFICAÇÃO DOS ELEMENTOS EM CASO DE HOUVER ALGUM FALTANDO EM ALGUMA COLUNA.

corr = train_dados_new.corr()
print(corr["Category"])	#VE-SE QUE NÃO HÁ NENHUMA CORRELAÇÃO FORTE ENTRE OS ELEMENTOS.