import random
import numpy as np
#import sparse as sparse3d
import cv2 as cv
from scipy.sparse import dok_matrix
from math import inf



class ReturnPatch(object):
    ''' 
    Random patch centered around target border (positive) or totally random (negative).
    '''
    def __init__(self, ppositive, patch_size, kernel_shape=(3, 3), fullrandom=False, anyborder=True, debug=False,
                 segmentation=True, reset_seed=False, return_image_only=False):
        '''
        ppositive: chance of returning a patch centered around a point of the border
        patch_size: can be 2D or 3D
        kernel_shape: the higher the kernel (always 2D) the larger the border (more center candidates)
        fullrandom: Ignore positive/negative idea, just return a random patch
        anyborder: use for multi-channel targets
        debug: print some debug information
        segmentation: wether the target is a mask (True) or a class (False)
        reset_seed: wether to reset random seed in every call (creates more randomness when using multiple DataLoader workers)
        '''
        self.reset_seed = reset_seed
        self.psize = patch_size
        self.ppositive = ppositive
        self.return_image_only = return_image_only
        self.kernel = np.ones(kernel_shape, np.uint8)
        self.ks = kernel_shape
        self.fullrandom = fullrandom
        self.anyborder = anyborder
        self.debug = debug
        dim = len(patch_size)
        assert dim in (2, 3), "only support 2D or 3D patch"
        if dim == 3:
            self.volumetric = True
        elif dim == 2:
            self.volumetric = False
        self.segmentation = segmentation

    def random_choice_3d(self, keylist):
        '''
        Returns random point in 3D sparse COO object
        '''
        lens = [len(keylist[x]) for x in range(3)]
        assert lens[0] == lens[1] and lens[0] == lens[2] and lens[1] == lens[2], "error in random_choice_3d sparse matrix"
        position = random.choice(range(len(keylist[0])))
        point = [keylist[x][position] for x in range(3)]
        return point

    def __call__(self, image, mask=None, debug=False):
        '''
        Returns patch of image and mask
        '''
        debug = self.debug
        if self.reset_seed:
            random.seed()
        # Get list of candidates for patch center
        e2d = False
        shape = image.shape
        if not self.volumetric and len(shape) == 3:
            shape = (shape[1], shape[2])
            e2d = True
        #mask = np.asarray(mask)
        #import torch
        #mask = torch.from_numpy(mask.float())
        #print(mask.shape)
        #mask = mask.numpy()
        #print(mask.dtype)
        if not self.fullrandom:
            if self.volumetric:
                if mask.ndim > 3:
                    if self.anyborder is True:
                        hmask = mask.sum(axis=0)
                    else:
                        hmask = mask[self.anyborder]
                else:
                    hmask = mask
                borders = np.zeros(shape, dtype=mask.dtype)
                for i in range(shape[0]):
                    uintmask = (hmask[i]*255).astype(np.uint8)
                    borders[i] = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(mask.dtype)
                sparse = sparse3d.COO.from_numpy(borders)
                keylist = sparse.nonzero()
            else:
                if mask.ndim > 2:
                    if self.anyborder is True:
                        hmask = mask.sum(axis=0)
                    else:
                        hmask = mask[self.anyborder]
                else:
                    hmask = mask
                # Get border of mask
                uintmask = (hmask*255).astype(np.uint8)
                borders = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(hmask.dtype)
                sparse = dok_matrix(borders)
                keylist = list(sparse.keys())
                if debug:
                    print("Candidates {}".format(keylist))

        # Get top left and bottom right of patch centered on mask border
        four_d_volume = int(image.ndim == 4)
        if self.segmentation:
            four_d_mask = int(mask.ndim == 4)

        tl_row_limit = shape[0 + four_d_volume] - self.psize[0]
        tl_col_limit = shape[1 + four_d_volume] - self.psize[1]
        if self.volumetric:
            tl_depth_limit = shape[2 + four_d_volume] - self.psize[2]
            tl_rdepth = inf
        tl_rrow = inf
        tl_rcol = inf

        if self.fullrandom:
            if self.volumetric:
                tl_rrow, tl_rcol, tl_rdepth = (random.randint(0, tl_row_limit), random.randint(0, tl_col_limit),
                                               random.randint(0, tl_depth_limit))
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)
        elif len(keylist[0]) > 0 and random.random() < self.ppositive:
            if self.volumetric:
                while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit or tl_rdepth > tl_depth_limit:
                    tl_rrow, tl_rcol, tl_rdepth = self.random_choice_3d(keylist)
                    tl_rrow -= self.psize[0]//2
                    tl_rcol -= self.psize[1]//2
                    tl_rdepth -= self.psize[2]//2
            else:
                while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit:
                    tl_rrow, tl_rcol = random.choice(list(sparse.keys()))
                    tl_rrow -= self.psize[0]//2
                    tl_rcol -= self.psize[1]//2
        else:
            if self.volumetric:
                tl_rrow, tl_rcol, tl_rdepth = (random.randint(0, tl_row_limit), random.randint(0, tl_col_limit),
                                               random.randint(0, tl_depth_limit))
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)

        if tl_rrow < 0:
            tl_rrow = 0
        if tl_rcol < 0:
            tl_rcol = 0
        if self.volumetric:
            if tl_rdepth < 0:
                tl_rdepth = 0

        if debug:
            print("Patch top left(row, col): {} {}".format(tl_rrow, tl_rcol))

        if self.volumetric:
            if four_d_volume:
                rimage = image[:, tl_rrow:tl_rrow + self.psize[0],
                               tl_rcol:tl_rcol + self.psize[1],
                               tl_rdepth:tl_rdepth + self.psize[2]]
            else:
                rimage = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1], tl_rdepth:tl_rdepth + self.psize[2]]

            if self.segmentation:
                if four_d_mask:
                    rmask = mask[:, tl_rrow:tl_rrow + self.psize[0],
                                 tl_rcol:tl_rcol + self.psize[1],
                                 tl_rdepth:tl_rdepth + self.psize[2]]
                else:
                    rmask = mask[tl_rrow:tl_rrow + self.psize[0],
                                 tl_rcol:tl_rcol + self.psize[1],
                                 tl_rdepth:tl_rdepth + self.psize[2]]
            else:
                rmask = mask
        else:
            if e2d:
                rimage = image[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                rimage = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]

            if self.segmentation:
                if len(mask.shape) > 2:
                    rmask = mask[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
                else:
                    rmask = mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                rmask = mask

        self.tl_rrow = tl_rrow
        self.tl_rcol = tl_rcol
        if self.volumetric:
            self.tl_rdepth = tl_rdepth

        if rmask is None and self.return_image_only:
            return rimage
        else:
            return rimage, rmask

    def __str__(self):
        return (f"ReturnPatch: ppositive {self.ppositive}, patch_size {self.psize}, kernel_shape {self.ks}, volumetric {self.volumetric}, "
                f"anyborder {self.anyborder}, fullrandom {self.fullrandom}, segmentation {self.segmentation}, reset_seed {self.reset_seed}")
