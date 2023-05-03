import os
import csv
import json

def process_csv_file(csv_file_path):
    """
    Função auxiliar para ler o arquivo CSV e retornar um dicionário com as chaves
    sendo os nomes das colunas e os valores sendo uma lista com os valores de cada
    coluna.
    """
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        result = {col_name: [] for col_name in header}
        for row in reader:
            for i, value in enumerate(row):
                result[header[i]].append(value)
        return result

def process_directory(directory_path):
    """
    Função para processar um diretório, retornando um dicionário com as chaves
    sendo os nomes dos arquivos .csv encontrados e os valores sendo um dicionário
    com as colunas e valores do arquivo.
    """
    result = {}
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            result[file_name[:-4]] = process_csv_file(file_path)
    return result

def process_subjects(root_directory):
    """
    Função para processar todos os sujeitos (pastas com nome numérico) dentro do
    diretório raiz e retornar uma lista de dicionários com os dados processados.
    """
    result = []
    for subject_name in os.listdir(root_directory):
        subject_path = os.path.join(root_directory, subject_name)
        if os.path.isdir(subject_path):
            inCCsight_path = os.path.join(subject_path, 'inCCsight')
            if os.path.isdir(inCCsight_path):
                subject_data = process_directory(inCCsight_path)
                result.append({subject_name: subject_data})
    return result

if __name__ == '__main__':
    root_directory = '/home/jovi/Dados/fast'
    subjects_data = process_subjects(root_directory)
    json_data = json.dumps(subjects_data)

    output_directory = '/home/jovi/Documentos'
    output_file_path = os.path.join(output_directory, 'output.json')

    with open(output_file_path, 'w') as output_file:
        output_file.write(json_data)
