# coding: utf-8

def boundaries(imagem, conn = 8, dir = 'cw'):
    import numpy as np

    imagem = np.array(imagem, dtype='int32')

    bordasuperior = (imagem-imagem[np.r_[0,0:imagem.shape[0]-1],:])==1
    bordainferior = (imagem-imagem[np.r_[1:imagem.shape[0],imagem.shape[0]-1],:])==1
    bordadireita = (imagem-imagem[:,np.r_[1:imagem.shape[1],imagem.shape[1]-1]])==1
    bordaesquerda = (imagem-imagem[:,np.r_[0,0:imagem.shape[1]-1]])==1
    borda = bordasuperior+bordainferior+bordadireita+bordaesquerda

    y,x = np.nonzero(borda)
    boundary = np.zeros((x.shape[0],2))
    deslocamentoy = np.array([-1,-1,0,1,1,1,0,-1])
    deslocamentox = np.array([0,1,1,1,0,-1,-1,-1])
    vizinho = 1
    ybase = y[0]
    xbase = x[0]
    indice = 0
    boundary[indice] = [ybase,xbase]

    for i in np.arange(x.shape[0]-1):
        while (borda[ybase+deslocamentoy[vizinho],xbase+deslocamentox[vizinho]]==0):
            
            vizinho = (vizinho+1)%8
        indice = indice+1
        ybase = ybase+deslocamentoy[vizinho]
        xbase = xbase+deslocamentox[vizinho]
        boundary[indice] = [ybase,xbase]
        vizinho = (vizinho+5)%8
    return boundary

def points(resultado,npoints):
    import numpy as np
    from scipy import interpolate
    from scipy.ndimage.morphology import binary_closing

    #boundary = np.array([[2,1],[1,2],[1,3],[2,4],[2,5],[3,5],[3,4],[3,3],[2,2],[3,1]])

    ##### Determinação dos pontos extremos do corpo caloso #####

    boundary = boundaries(resultado)
    threshold = boundary[:,1].mean()
    frente = boundary[boundary[:,1]<threshold,:]
    fundo = boundary[boundary[:,1]>=threshold,:]
    yfrente = frente[:,0].max()
    xfrente = frente[frente[:,0]==yfrente,1].max()
    candfrente = np.nonzero(boundary[:,0]==yfrente)[0]

    try:
        indfrente = candfrente[boundary[boundary[:,0]==yfrente,1]==xfrente]
        yfundo = fundo[:,0].max()
        xfundo = np.floor(np.median(fundo[fundo[:,0]==yfundo,1]))
        candfundo = np.nonzero(boundary[:,0]==yfundo)[0]
        indfundo = candfundo[boundary[boundary[:,0]==yfundo,1]==xfundo]
        indfrente = indfrente[0]
        indfundo = indfundo[0]
    except:
        indfrente = candfrente[boundary[boundary[:,0]==yfrente,1]==xfrente]
        yfundo = fundo[:,0].max()
        xfundo = np.floor(np.median(fundo[fundo[:,0]==yfundo,1]))
        candfundo = np.nonzero(boundary[:,0]==yfundo)[0]
        indfundo = candfundo[boundary[boundary[:,0]==yfundo,1]==xfundo-1]
        indfrente = indfrente[0]
        indfundo = indfundo[0]        

    ##### Determinação das bordas superior e inferior #####
    if (indfrente > indfundo):
        bounddown = boundary[indfrente:indfundo-1:-1,:]
        boundup = boundary[np.r_[indfrente:boundary.shape[0],:indfundo+1],:]
    else:
        boundup = boundary[indfrente:indfundo+1,:]
        bounddown = boundary[np.r_[indfrente:-1:-1,boundary.shape[0]-1:indfundo-1:-1],:]

    unew = np.linspace(0,1,npoints)
    tck,u = interpolate.splprep(boundup.transpose(),s=0)
    yupInter,xupInter = interpolate.splev(unew,tck)
    tck,u = interpolate.splprep(bounddown.transpose(),s=0)
    ydownInter,xdownInter = interpolate.splev(unew,tck)

    ymedio = (yupInter+ydownInter)/2
    xmedio = (xupInter+xdownInter)/2

    py = ymedio
    px = xmedio


    return px, py

def thickness(resultado, npoints):
    import numpy as np
    from scipy import interpolate

    boundary = boundaries(resultado)
    #boundary = np.array([[2,1],[1,2],[1,3],[2,4],[2,5],[3,5],[3,4],[3,3],[2,2],[3,1]])

    ##### Determinação dos pontos extremos do corpo caloso #####

    threshold = boundary[:,1].mean()
    frente = boundary[boundary[:,1]<threshold,:]
    fundo = boundary[boundary[:,1]>=threshold,:]
    yfrente = frente[:,0].max()
    xfrente = frente[frente[:,0]==yfrente,1].max()
    candfrente = np.nonzero(boundary[:,0]==yfrente)[0]

    try:
        indfrente = candfrente[boundary[boundary[:,0]==yfrente,1]==xfrente]
        yfundo = fundo[:,0].max()
        xfundo = np.floor(np.median(fundo[fundo[:,0]==yfundo,1]))
        candfundo = np.nonzero(boundary[:,0]==yfundo)[0]
        indfundo = candfundo[boundary[boundary[:,0]==yfundo,1]==xfundo]
        indfrente = indfrente[0]
        indfundo = indfundo[0]
    except:
        indfrente = candfrente[boundary[boundary[:,0]==yfrente,1]==xfrente]
        yfundo = fundo[:,0].max()
        xfundo = np.floor(np.median(fundo[fundo[:,0]==yfundo,1]))
        candfundo = np.nonzero(boundary[:,0]==yfundo)[0]
        indfundo = candfundo[boundary[boundary[:,0]==yfundo,1]==xfundo-1]
        indfrente = indfrente[0]
        indfundo = indfundo[0]    

    ##### Determinação das bordas superior e inferior #####
    if (indfrente > indfundo):
        bounddown = boundary[indfrente:indfundo-1:-1,:]
        boundup = boundary[np.r_[indfrente:boundary.shape[0],:indfundo+1],:]
    else:
        boundup = boundary[indfrente:indfundo+1,:]
        bounddown = boundary[np.r_[indfrente:-1:-1,boundary.shape[0]-1:indfundo-1:-1],:]

    unew = np.linspace(0,1,npoints)
    tck,u = interpolate.splprep(boundup.transpose(),s=0)
    yupInter,xupInter = interpolate.splev(unew,tck)
    tck,u = interpolate.splprep(bounddown.transpose(),s=0)
    ydownInter,xdownInter = interpolate.splev(unew,tck)

    pts_up = np.vstack((xupInter, yupInter))
    pts_dw = np.vstack((xdownInter, ydownInter))

    thickness = np.linalg.norm((pts_up-pts_dw), axis=0)

    return thickness, pts_up, pts_dw
