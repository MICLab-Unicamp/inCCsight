# InCCsight

O inCCsight é uma ferramenta desktop open source para processar, explorar e analisar dados do Corpo Caloso (CC). 

## Considerations || Corrigir antes de subir
This software was developed through the [MICLab](https://miclab.fee.unicamp.br/), check out our [Github](https://github.com/MICLab-Unicamp)!

**Article:** The article that explains the development and operation of the tool was published by [Computer & Graphics](https://www.journals.elsevier.com/computers-and-graphics). 
You can check out this article [here](https://www.sciencedirect.com/science/article/abs/pii/S0097849321001436).
In case of using the software, please cite this article: 

*Caldeira, Thais, et al. "inCCsight: A software for exploration and visualization of DT-MRI data of the Corpus Callosum." Computers & Graphics 99 (2021): 259-271.*

**Data**: In case you want to test the tool and do not have the data (DTI), check the [OASIS](https://www.oasis-brains.org/#access). This is a data center with medical images available for studies and collaboration of science community. 

If you use data from Oasis, check out the Oasis3 notebook in this repository, it performs a pre-processing of data collected from the data center.

For an overview of the tool, we have a video showing the process and use: [InCCsight](https://www.youtube.com/watch?v=9Y8s8H3X2ow&list=PLCZ64jtDHDO0fBxdyRM5jtukD3U_ZxME_&index=3)

## Instalação

### Website
Para instalar a partir do Website, basta seguir os passos:
1. Acessar o website: [https://miclab.fee.unicamp.br](MICLab)
2. Aba de Download e selecionar o seu sistema operacional.
3. Clique em baixar.


### Da fonte
Para instalar a ferramenta diretamente a partir da fonte (Github) e ter acesso aos códigos:
1. No terminal, digite `git clone https://github.com/MICLab-Unicamp/inCCsight`
2. Caminhe até o local que foi realizado o download e realize o comando: `npm install`
3. **Tipos de Uso:** 
  - 3.1 **Usuário:** Use o comando `yarn electron:build`. Isso irá transformar a ferramenta em um executável Windows, Linux e Mac.
  - 3.2. **Desenvolvedor:** use o comando `yarn electron:serve`. Isso iniciará a ferramenta como desenvolvedor, permitindo realizar alterações no código e visualiza-las em tempo real.


Após a instalação, a primeira vez que abrir a ferramenta pode demorar um pouco para a inicialização, pois o anti-virus verificará a segurança.

## Uso

## Como Criar um Novo Método
Para criar um novo método dentro do Software, o usuário deve:
- Possuir uma pasta com o nome do método.
- Um arquivo "main.py" dentro da pasta.
- O arquivo "main.py" deve ter o seguinte código para leitura dos dados:
  ```python
    import argparse
    import glob
    import os
    import your_segmentation_method

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parent', nargs='*', dest='parents')

    args = parser.parse_args()
        
    # Read files
    folder_mri = args.parents

    all_subjects = []

    for folder in folder_mri:
        subjects = glob.glob(os.path.join(folder, "*"))

        for subject in subjects:
            all_subjects.append(subject)

    your_segmentation_method.segment(all_subject)
  ```
- O seu método deve receber uma lista com o caminho de cada sujeito a ser processado.
- Para que os dados sejam exibidos nos gráficos e tabelas da interface, o código deve gerar um arquivo CSV na pasta "../methods/csv"


## Exemplo

## Contribuição

## Contato

## Licença

## Avisos 

No Linux, em package.json, adicionar os scripts:
    "start": "react-scripts --openssl-legacy-provider start",
    "build": "react-scripts --openssl-legacy-provider build"

No Windows:
    "start": "react-scripts start",
    "build": "react-scripts build"