from torchvision import transforms
import torch
import random
import numpy as np

from return_patch import ReturnPatch



class SegmentationTransform():
    '''
    Applies a torchvision transform into image and segmentation target
    '''
    def __init__(self, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform
    
    def __call__(self, image, seg_image):
        '''
        Precisa-se fixar a seed para mesma transformada ser aplicada
        tanto na máscara quanto na imagem. 
        '''
        # Gerar uma seed aleatória
        seed = np.random.randint(2147483647)

        # Fixar seed e aplicar transformada na imagem
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.transform is not None:
            image = self.transform(image)
            
        # # Fixar seed e aplicar transformada na máscara
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.target_transform is not None:
            seg_image = self.target_transform(seg_image)
        
        #if self.transform is not None:
        #    seg_image = seg_image.numpy()
        #    image, seg_image = self.transform(image, seg_image)

        return image, seg_image
    
    def __str__(self):
        return f"SegmentationTransform: {self.transform}/{self.target_transform}"
        
def get_transform(string): 
    '''
    Mapeamento de uma string a uma transformadas.
    '''
    if string is None:
        image_transform = None
        target_transform = None    
    elif string == "RandomCrop":
        return ReturnPatch(None, (88, 64, 88), fullrandom=True)
    else:
        raise ValueError(f"{string} does not correspond to a transform.")
    
    return SegmentationTransform(image_transform, target_transform)
