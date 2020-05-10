"""
Shape signature profile Module
"""
import numpy as np
import scipy.interpolate as spline
import scipy.ndimage as nima

def sign_extract(seg, resols, smoothness, points): #Function for shape signature extraction
    splines = get_spline(seg,smoothness)

    sign_vect = np.array([]).reshape(0,points) #Initializing temporal signature vector
    for resol in resols:
        sign_vect = np.vstack((sign_vect, get_profile(splines, n_samples=points, radius=resol)))

    return sign_vect

def sign_fit(sig_ref, sig_fit, points): #Function for signature fitting
    dif_curv = []
    for shift in range(points):
        dif_curv.append(np.abs(np.sum((sig_ref - np.roll(sig_fit[0],shift))**2)))
    return np.apply_along_axis(np.roll, 1, sig_fit, np.argmin(dif_curv))

def compute_angles(pivot, anterior, posterior):
    max_angle = np.pi*2

    def angles(vectors):
        return np.arctan2(vectors[1], vectors[0])

    ap, pp = anterior-pivot, posterior-pivot
    ang_post, ang_ant = angles(pp), angles(ap)
    ang = ang_post - ang_ant

    dif_prof = np.abs(ang - np.roll(ang,1)) > np.pi
    ind_start = np.where(dif_prof)[0][::2]
    ind_end = np.where(dif_prof)[0][1::2]
    zeros = np.zeros_like(ang)
    for in1, in2 in zip(ind_start,ind_end):
        if (ang[in1] - np.roll(ang,1)[in1]) > np.pi:
            zeros[in1:in2] = -2*np.pi
        else:
            zeros[in1:in2] = 2*np.pi
    return (ang + zeros) *180/(np.pi)

def get_profile(tck, n_samples, radius):
    def eval_spline(tck, t):
        y, x = spline.splev(t,tck)
        return np.vstack((y,x))

    t_pivot = np.linspace(0,1, n_samples, endpoint=False)
    pivot = eval_spline(tck, t_pivot)
    t_anterior = np.mod(t_pivot+(1-radius), 1)
    anterior = eval_spline(tck, t_anterior)
    t_posterior = np.mod(t_pivot+radius, 1)
    posterior = eval_spline(tck, t_posterior)

    return compute_angles(pivot, anterior, posterior)

def get_seq_graph(edge):

    dy, dx = np.array([-1,0,1,1,1,0,-1,-1]), np.array([-1,-1,-1,0,1,1,1,0])
    def get_neighbors(node):
        Y, X = node[0]+dy, node[1]+dx
        neighbors = edge[Y, X]
        Y, X = Y[neighbors], X[neighbors]
        return list(zip(Y,X))
    graph = {}
    Y, X = edge.nonzero()
    for node in zip(Y,X):
        graph[node] = get_neighbors(node)
    seq = []
    first_el = (Y[0], X[0])
    seq.append(first_el)
    ext_el = first_el
    act_el = graph[ext_el][0]
    while (first_el != ext_el) or (len(seq)==1):
        ind_el = np.where(np.array(graph[(ext_el)])!=act_el)
        ind_el_uq = np.unique(ind_el[0])

        if len(ind_el_uq)==1:
            ind_el = ind_el_uq[0]
        else:
            acum_dist = []
            for ind in ind_el_uq:
                dist_ = (graph[ext_el][ind][0]-ext_el[0])**2+(graph[ext_el][ind][1]-ext_el[1])**2
                acum_dist.append(dist_)
            min_dist = acum_dist.index(min(acum_dist))
            ind_el = ind_el_uq[min_dist]

        act_el = ext_el
        ext_el = graph[(act_el)][ind_el]
        seq.append(ext_el)
    lst1, lst2 = zip(*seq)

    return (np.array(lst1), np.array(lst2))

def get_spline(seg,s):
    nz = np.nonzero(seg)
    x1,x2,y1,y2 = np.amin(nz[0]),np.amax(nz[0]),np.amin(nz[1]),np.amax(nz[1])
    M0 = seg[x1-5:x2+5,y1-5:y2+5]
    nescala = [4*M0.shape[-2],4*M0.shape[-1]]
    M0 = resizedti(M0,nescala).astype('bool')
    M0_ero = nima.binary_erosion(M0).astype(M0.dtype)
    con_M0 = np.logical_xor(M0_ero,M0)
    seq = get_seq_graph(con_M0)
    tck, _ = spline.splprep(seq, k=5, s=s)

    return tck

def resizedti(img,shape):

    y,x = np.indices(shape)
    x = x/(shape[1]/img.shape[-1])
    y = y/(shape[0]/img.shape[-2])
    return img[y.astype('int'),x.astype('int')]