import os
import numpy as np
#import torch
from glob import iglob
from torch.utils.data import Dataset


MODES = ["train", "val"]


class DatasetMRI(Dataset):
    '''
    A herança da classe Dataset definida pelo torch garante compatibilidade 
    com utilidades de dataset implementadas pelo torch.
    '''
    def __init__(self, mode, folder, transform=None, debug=False):
        '''
        mode: train, val or test
        folder: pasta em que as imagens estão
        transform: transformadas a serem aplicadas nas imagens
        '''
        self.folder = folder
        self.transform = transform
        self.debug = debug
        
        all_images = sorted(list(iglob(os.path.join(folder, mode, "**", "*_FA.npz"), recursive=True)))        

        self.dataset = all_images
        print('Tamanho do Datset:',len(self.dataset))

        print(f"{mode} DatasetMRI. Localização das Imagens: {self.folder}. Transforms: {self.transform}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        '''
        Pega informações indexadas na posição 'i' e faz o trabalho de conversões 
        e transformadas, retornando os dados prontos para treino.
        '''
        # Carregando dados do .npz como tensores torch 
        npz_file = self.dataset[i]
        npz = np.load(npz_file)

        dirname = os.path.basename(self.folder)

        if dirname in ["Dataset_CCexp", "Dataset_HCManualCCexp", "Dataset_HCexpCC"]:
            image = npz["dataFA"][:]
            image = image.astype(np.float32)
            seg_image = npz["mask"].astype(np.float32)
            #print(image.shape, seg_image.shape)
            #x, y, z = size_pad(image)
            #print(x,y,z)
            image = np.expand_dims(image, 0)
            seg_image = np.expand_dims(seg_image, 0)
            #image, seg_image = image.unsqueeze(0), seg_image.unsqueeze(0)

#            subject = tio.Subject(
#                image=tio.ScalarImage(tensor = image),
#                seg_image=tio.LabelMap(tensor = seg_image),
#            )

#            transform = tio.Pad((10))
#            transform = tio.Compose([
                                    #tio.Pad((10)),
#                                    tio.Pad((x, y, z)),
                                    #tio.Pad((10,10,100,100,200,200)),
#                                    ])
#            transformed = transform(subject)
#            image = transformed.image.data.numpy()
#            seg_image = transformed.seg_image.data.numpy()
            #print(image.shape, seg_image.shape)

        elif dirname == "dados_teste_3D":
            image_FA = npz["dataFA"][:]
            image_MD = npz["dataMD"][:]
            image_MO = npz["dataMO"][:]
            image = np.stack([image_FA, image_MD, image_MO]).astype(np.float32)
            #print(image.shape)
            seg_image = npz["mask"].astype(np.float32)
            seg_image = np.expand_dims(seg_image, 0)

        else:
            raise ValueError(f"self.folder {dirname} errado")

        if self.transform is not None:
            image, seg_image = self.transform(image, seg_image)

        return_dict = {"image": image, "seg_image": seg_image}

        return return_dict

