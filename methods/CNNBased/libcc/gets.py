# coding: utf-8

def getTheCC(segmentation):

    import numpy as np
    from skimage.measure import label, regionprops

    labels = label(input=segmentation, neighbors=4)
    regions = regionprops(labels)

    theCC = []
    maxwidth = 0
    i = 1

    ymed = None
    xmed = None

    # background is labeled as 0
    for props in regions[1:]:
        minr, minc, maxr, maxc = props.bbox                            
        dx = maxc-minc
        dy = maxr-minr

        if dx > maxwidth:
            maxwidth = dx
            if maxr < 60:
                theCC = labels == i+1
                ymed = maxr-dy/2
                xmed = maxc-dx/2
        i=i+1
                
    return theCC, ymed, xmed

def getCentralPoint(kmeans, k, kpoints):

    import numpy as np

    # kmeans = result of cluster.Kmeans().fit_predict()
    kcenters = []
    for i in range(0,k):
      
      # Get min and max idx from same label
        min_idx = np.min(np.where(kmeans == i))
        max_idx = np.max(np.where(kmeans == i))
      
        # Middle term idx
        mid_idx = min_idx + (max_idx-min_idx)/2

        # Get value using idx
        xk = int(round(kpoints[mid_idx][0]))
        yk = int(round(kpoints[mid_idx][1]))

        kcenters.append([xk,yk])
    
    return kcenters

def getGroupPoints(kmeans, k, offset, kpoints):

    import numpy as np

    # kmeans = result of cluster.Kmeans().fit_predict()
    # offset = how many points in the borders will be ignored
    
    kcenters = []
    for i in range(0, k):
      
        # Get min and max idx from same label
        min_idx = np.min(np.where(kmeans == i))
        max_idx = np.max(np.where(kmeans == i))
      
        for j in range(min_idx + offset, max_idx - offset):
        
            # Get value using idx
            xk = kpoints[j][0]
            yk = kpoints[j][1]

            kcenters.append([i,xk,yk])
    
    return kcenters

def getScalars(segm, wFA, wMD, wRD, wAD):

    import numpy as np    

    # Total value
    meanFA = np.mean(wFA[segm==True])
    stdFA = np.std(wFA[segm==True])

    meanMD = np.mean(wMD[segm==True])
    stdMD = np.std(wMD[segm==True])

    meanRD = np.mean(wRD[segm==True])
    stdRD = np.std(wRD[segm==True])

    meanAD = np.mean(wAD[segm==True])
    stdAD = np.std(wAD[segm==True])
    
    return meanFA, stdFA, meanMD, stdMD, meanRD, stdRD, meanAD, stdAD

def getFAmidline(segm, wFA_ms, n_points=200):

    import numpy as np
    from libcc import points

    # Get CC's midline
    px, py = points(segm, n_points+1)

    fa_line = []
    for aux in range(0, n_points):
        try:
            x = int(round(px[aux]))
            y = int(round(py[aux]))
            fa = wFA_ms[y,x]
        except:
            x = int(np.floor(px[aux]))
            y = int(np.floor(py[aux]))
            fa = wFA_ms[y,x]
        fa_line.append(fa)

    return fa_line	

def getData(parcel, FA, MD, RD, AD):
    
    import numpy as np
    
    data = {}

    # Initialize
    for region in ['P1', 'P2', 'P3', 'P4', 'P5']:
        data[region] = {}

    # Parcel values
    for i in range(2,7):
        data['P'+str(i-1)]['FA'] = np.mean(FA[parcel==i])
        data['P'+str(i-1)]['FA StdDev'] = np.std(FA[parcel==i])

        data['P'+str(i-1)]['MD'] = np.mean(MD[parcel==i])
        data['P'+str(i-1)]['MD StdDev'] = np.std(MD[parcel==i])

        data['P'+str(i-1)]['RD'] = np.mean(RD[parcel==i])
        data['P'+str(i-1)]['RD StdDev'] = np.std(RD[parcel==i])

        data['P'+str(i-1)]['AD'] = np.mean(AD[parcel==i])
        data['P'+str(i-1)]['AD StdDev'] = np.std(AD[parcel==i])

    return data
	

def getLargestConnectedComponent(segmentation):

    from skimage.measure import label, regionprops  
    import numpy as np

    labels = label(segmentation)
    assert(labels.max() != 0 ) # assume at least 1 CC
    cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return cc