import cc3d
import numpy as np
from operator import itemgetter
import torch

def get_post_processed_cc3d(volume_saida_bin):
    '''
    volume: input volume
    verbose: prints label_count
    returns:
        filtered_volume, label_count, labeled_volume
    '''
    labels_out = cc3d.connected_components(volume_saida_bin.cpu().numpy().astype(np.int32))
    label_count = np.unique(labels_out, return_counts=True)[1]
    label_count = [(label, count) for label, count in enumerate(label_count)]
    label_count.sort(key=itemgetter(1), reverse=True)
    label_count.pop(0)  

    id_max = label_count[0][0]
    filtered = labels_out == id_max
    filtered = torch.tensor(filtered).cpu()
    volume = filtered * volume_saida_bin.cpu()
    return volume