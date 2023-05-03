# Bibliotecas
import json
import ast
import pandas as pd

# Classe Subject

class Subject():
  def __init__(self, name, watershed_scalar, ROQS_scalars, watershed_midlines, ROQS_midlines, 
               watershed_thickness, ROQS_thickness, watershed_parcellation_statistics, ROQS_parcellation_statistics, santarosa_scalars):
    self.name = self.adjust_name(str(name))
    self.watershed_scalars = watershed_scalar 
    self.ROQS_scalars = ROQS_scalars
    self.watershed_midlines = watershed_midlines
    self.ROQS_midlines = ROQS_midlines
    self.watershed_thickness = list(watershed_thickness)
    self.ROQS_thickness = list(ROQS_thickness)
    self.watershed_parcellation = watershed_parcellation_statistics
    self.ROQS_parcellation = ROQS_parcellation_statistics
    self.santarosa_scalars = santarosa_scalars

  def adjust_name(self, name):

    while len(name) != 7:
      name = f"0{name}"

    return name

  def create_json(self):
    subject = {
              "Id": self.name,
               
               # Os valores escalares de segmentação
              "Watershed_scalar": dict(self.watershed_scalars),
              "ROQS_scalar": dict(self.ROQS_scalars),
              "santarosa_scalars": dict(self.santarosa_scalars),

              # Midlines
               "Watershed_midlines": dict(self.watershed_midlines),
               "ROQS_midlines": dict(self.ROQS_midlines),

              # Thickness
               "Watershed_thickness": self.watershed_thickness,
               "ROQS_thickness": self.ROQS_thickness,
              
              # Parcellation
               "Watershed_parcellation": dict(self.watershed_parcellation),
               "ROQS_parcellation": dict(self.ROQS_parcellation)

               }
    return subject

# Criar uma função que remove os sujeitos com falhas 
import math

def checkSubject(subject):
  keys = subject.keys()
  for key in keys:
    if type(subject[key]) == dict:
      keys_2 = subject[key].keys()
      for key_2 in keys_2:
        if type(subject[key][key_2]) == list:
          for i in range(0, len(subject[key][key_2])):
            if math.isnan(subject[key][key_2][i]) == True:
              return True
        else:
          if(math.isnan(subject[key][key_2]) == True):
              return True

def removeSubjects(subjects_list):
  faileds = []
  for i in range(0, len(subjects_list)):
    flag = checkSubject(subjects_list[i])
    if flag == True:
      faileds.append(i)
  return faileds

def transformInJson(data):
  with open("../../src/data/mydata.json", "w") as final:
    json.dump(data, final)

def dataFrameStringToList(df):
  columns = df.columns
  for i in range(0, len(df)):
    for column in columns:
      df.iloc[i][column] = ast.literal_eval(df.iloc[i][column])
  return df

# Importando e executando
watershed_scalar = pd.read_csv("Watershed_scalar_statistics.csv", sep=";").drop(["Unnamed: 0"], 1)
ROQS_scalar = pd.read_csv("ROQS_scalar_statistics.csv", sep=";").drop(["Unnamed: 0"], 1)
santarosa_scalar = pd.read_csv("santarosa.csv", sep=";").drop(["Unnamed: 0"], 1)

watershed_midlines = pd.read_csv("Watershed_scalar_midlines.csv", sep=";").drop(["Unnamed: 0"], 1)
ROQS_midlines = pd.read_csv("ROQS_scalar_midlines.csv", sep=";").drop(["Unnamed: 0"], 1)

watershed_midlines = dataFrameStringToList(watershed_midlines)
ROQS_midlines = dataFrameStringToList(ROQS_midlines)

watershed_thickness = pd.read_csv("Watershed_dict_thickness.csv", sep=";").drop(["Unnamed: 0"], 1)
ROQS_thickness = pd.read_csv("ROQS_dict_thickness.csv", sep=";").drop(["Unnamed: 0"], 1)

watershed_parcellation_statistics = pd.read_csv("Watershed_parcellation_statistics.csv", sep=";").drop(["Unnamed: 0"], 1)
ROQS_parcellation_statistics = pd.read_csv("ROQS_parcellation_statistics.csv", sep=";").drop(["Unnamed: 0"], 1)

names = list(ROQS_parcellation_statistics["Name"])

subjects_list = []
for i in range(0, len(names)):
  sub = Subject(names[i], watershed_scalar.iloc[i], ROQS_scalar.iloc[i], watershed_midlines.iloc[i], ROQS_midlines.iloc[i], watershed_thickness.iloc[i], ROQS_thickness.iloc[i], watershed_parcellation_statistics.iloc[i], ROQS_parcellation_statistics.iloc[i], santarosa_scalar.iloc[i])
  sub_json = sub.create_json()
  subjects_list.append(sub_json)

for j in range(0, 5):
  faileds = removeSubjects(subjects_list)  
  try:
    for i in faileds:
      subjects_list.pop(i)
  except:
    continue

transformInJson(subjects_list)