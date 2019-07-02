from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from scipy.misc import factorial
import numpy as np 
import math
import random
import numpy as np
import GPy
import matplotlib.pyplot as plt
import itertools


"""
Rather than rasterise the polygons this builds the hypercuboids without going back
to a raster, so it will probably be better for high dimensional problems.

However it's much slower for low-dim problems, as it's having to check a lot more
inside-polygon queries.
"""

#round (by probability)
#https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return int(sign * round_func(x))
    
def pointsinrec(ps,rec):
    return np.all((ps>rec[::2]) & (ps<rec[1::2]),1)

def sharededge(pgb1,pgb2):
    #TODO Should check these are two contiguous vertices
    return np.sum([np.any(np.all(pgb1==vert,1)) for vert in pgb2])>1 #two points that match = edge that matches

def test_inside_simplex(simplex,point):
    try:
        hull = Delaunay(simplex)
        return hull.find_simplex(point)>=0
    except QhullError:
        return False #the triangle's probably just flat
        
def test_inside_any_simplex(polygon,point,dims):
    """Given a 2d (NxD) array, consisting of vertex locations on each row, and each d+1 rows containing one simplex"""    
    for i in range(0,len(polygon),dims+1):
        if test_inside_simplex(polygon[i:i+dims+1,:], point):
            return True
    return False


def splitintopolygons(polygons,d):
    """
    Give a 1d array of length (N*(d*(d+1))), consisting of a series of N simplexes. Each simplex
    has d+1 locations, each location is d long. For example, for 2d:
                                  [(x1,y1),(x2,y2),(x3,y3)][(x1,y1),(x2,y2),(x3,y3)]...
    d = dimensions
    returns a list of arrays. Each array is n x d, i.e. a list of points. Each d+1 rows of the array is one simplex.    
    Each array is one polygon.
    """
    polygonboundaries = []
    for i,simplex in enumerate(polygons.reshape(int(len(polygons)/(d*(d+1))),(d*(d+1)))):
        polygonboundaries.append(simplex.reshape(d+1,d))
    change = 3
    i = 0
    while change>0:
        if i>=len(polygonboundaries): 
            i = 0
            change = change - 1
        if np.isnan(polygonboundaries[i][0,0]):
            del polygonboundaries[i]
        j = 0
        while True:
            if j==i: j+=1
            if j>=len(polygonboundaries):
                break
            if i>=len(polygonboundaries):
                break
            if sharededge(polygonboundaries[i],polygonboundaries[j]):
                #print(getvertexindices(polygonboundaries[i],polygonboundaries[j]))
                polygonboundaries[i] = np.r_[polygonboundaries[i],polygonboundaries[j]]
                del polygonboundaries[j]
                change = 2
            j+=1
        i+=1
    return polygonboundaries

def simplexVolume(vectors,input_space_dim):
    """Returns the volume of the simplex defined by the
    row vectors in vectors, e.g. passing [[0,0],[0,2],[2,0]]
    will return 2 (as this triangle has area of 2)"""
    assert vectors.shape[0]==input_space_dim+1, "For a %d dimensional space there should be %d+1 vectors describing the simplex" % (self.input_space_dim, self.input_space_dim)
    return np.abs(np.linalg.det(vectors[1:,:]-vectors[0,:]))/factorial(input_space_dim)

def polytopeVolume(polytope,input_space_dim):
    """
    Pass a 1d array as a list of simplexes, e.g. a square made of two triangles:
    [1,0,0,1,0,0, 1,0,0,1,1,1]
    """
    Ncoords = input_space_dim*(input_space_dim+1)
    vol = 0
    for i in range(0,len(polytope),Ncoords):
        vectors = polytope[i:(i+Ncoords)].reshape(input_space_dim+1,input_space_dim)
        if np.isnan(vectors[0,0]):
            break
        vol += simplexVolume(vectors,input_space_dim)
    return vol

def compute_rectangles(polygon,step,dims,Nrecs=10,Ntrials=10):
    """
    Given a 2d (NxD) array, consisting of vertex locations on each row, and each d+1 rows containing one simplex
    Computes a list of rectangles. Each element of the output is a 2xd array, with the top row being one
    corner of the rectangle, and the bottom the opposite corner.
    
    Nrecs = number of rectangles to use
    Ntrials = number of times we'll try to get the best rectangle (greedy)
    """
    recs = []
    maxits = 100000
    for it in range(Nrecs):
        bestrec = None
        bestsize = -1
        for it in range(Ntrials):
            found = np.array([-1000000.0]*dims)

            for attempt in range(maxits):
                #trying to find a point that's not in other rectangles but also inside the polygon
                found = np.min(polygon)+np.random.rand(dims)*(np.max(polygon,0)-np.min(polygon,0))
                found=np.round(found/step)*step                
                ok = test_inside_any_simplex(polygon,found,dims)
                if not ok:
                    continue
                
                lst = list(itertools.product([0, 1], repeat=dims))
                lst = np.array(lst)
                for otherrec in recs:
                    if pointsinrec(found,otherrec):
                        ok = False
                        break
                    if not ok: break
                if ok: break
            if attempt==maxits:
                break
                
                  
            rec = found[None,:].repeat(2,0)
            #rec[0,:]-=step/2
            #rec[1,:]+=step/2
            recsize = np.ones(rec.shape[1])
            #recs = [rec]
            #try expanding rectangle in dim

            
            #same each time! Could be copied.
            thingstotry = []
            for dim in np.arange(rec.shape[1]):
                for changedir in [0,1]:
                    thingstotry.append([dim,changedir])
            validrec = False
            while len(thingstotry)>0:
                thingi = np.random.randint(0,len(thingstotry))
                dim = thingstotry[thingi][0]
                changedir = thingstotry[thingi][1]
                
                #dim = np.random.randint(0,rec.shape[1])
                trialrec = rec.copy()
                #changedir = np.random.randint(0,2)
                if changedir==1: 
                    change = step
                else:
                    change = -step
                trialrec[changedir,dim]+= change
                #check all corners of hypercube lie inside a simplex
                lst = list(itertools.product([0, 1], repeat=dims-1))
                lst = np.array(lst)
                coords = np.c_[lst[:,0:dim],changedir*np.ones([len(lst),1]),lst[:,dim:]]
                inside_polygon = True
                outside_rectangles = True
                pointstocheck = trialrec[coords.astype(int),np.arange(0,dims)]
                #also check middle of rectangle
                pointstocheck = np.r_[pointstocheck,np.mean(pointstocheck,0)[None,:]]
                for ps in pointstocheck:
                    if not test_inside_any_simplex(polygon,ps,dims):
                        inside_polygon = False
                    #and check they're outside other rectangles
                    for otherrec in recs:
                        if pointsinrec(ps,otherrec):
                            outside_rectangles = False
                            continue
                    if not outside_rectangles:
                        continue
                    
                if not inside_polygon or not outside_rectangles:
                    del thingstotry[thingi]
                    continue
                rec = trialrec
                recsize[dim]+=1
                validrec = True
                
            if (np.prod(recsize)>bestsize) and (validrec):
                bestsize = np.prod(recsize)
                bestrec = rec
        if bestrec is not None:
            recs.append(bestrec)
    return recs


def gettrainingdata(Xrow,dims,Nrecs,step,Ntrials=50):
    """
    Pass one row of X (a list of simplexes, e.g. 0,1,1,0,1,1,0,1,1,0,0,0,2,3,3,2,3,3,2,3,3,2,2,2 = two squares)
    Returns a bunch of rectangles that fills them all.
    """
    #compute the sum of the squareroots of the volumes of the polytopes.
    
    
    polys = splitintopolygons(Xrow,2)
    
    npolys = 0
    vols = np.zeros(len(polys))    
    for i,poly in enumerate(polys):
        if np.isnan(poly[0,0]): continue
        #vols[i] = np.sqrt(polytopeVolume(poly.flatten(),2))
        vols[i] = (polytopeVolume(poly.flatten(),2))
        npolys+=1
    
    allrecs = []
    for v,poly in zip(vols,polys):
        if np.isnan(poly[0,0]): continue
        #we give each polygon one rectangle, so that leaves Nrecs-npolys rectangles to play with
        #so each polygon gets 1+(Nrec-npolys)*(vol/sum of vol)
        target = prob_round(1+(Nrecs-npolys)*v/np.sum(vols))
        #print(1+(Nrecs-npolys)*v/np.sum(vols))
        recs,ps,oldps = compute_rectangles(poly,step,2,Nrecs=target,Ntrials=Ntrials)
        allrecs.extend(recs)
    return allrecs, polys, np.sum(vols), ps, oldps

def compute_newX(X,Nrecs=10,step=0.025,Ntrials=10,dims=2):
    """
    Given X consisting of an array, each row of which is a list of simplexes, describing an input, e.g.
       0,1,1,0,1,1,0,1,1,0,0,0,2,3,3,2,3,3,2,3,3,2,2,2
       2,1,3,0,4,1,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n     (n=nan)
    This computes a newX describing a bunch of hypercuboids, which lie approximately in such a way to cover the inputs
    The newX has each hypercuboid on a new row. So the startindices let us know which rows are for which of the rows in X.
    The allvolscales output notes by what ratio we over (or under) estimate the true volume of each output type.
    Example output:
    newX
       0,1,0,1
       2,3,2,3
       2,0,4,1
    startindicies
       0,2,3
    allvolscales
       1,0.8
    """
    newX = []
    allpolys = []
    allrecs = []
    allvols = []
    allps = []
    alloldps = []
    for Xrow in X:
        recs,polys,vol,ps, oldps = gettrainingdata(Xrow,dims,Nrecs=Nrecs,step=step,Ntrials=Ntrials)
        newX.append(np.array([rec.T.flatten() for rec in recs]))
        allpolys.append(polys)
        allrecs.append(recs)
        allvols.append(vol)
        allps.append(ps)
        alloldps.append(oldps)

    startindices = np.r_[np.array([0]),np.cumsum([len(n) for n in newX])]
    approxvols = [getvolumes(recs) for recs in allrecs]
    allvolscales = np.array(allvols)/np.array(approxvols)        
    return newX, startindices,allvolscales, allpolys, allrecs



def getvolumes(recs):
    return np.sum([np.prod(np.abs(rec[1::2]-rec[0::2])) for rec in recs])

def plotpolys(allpolys,allrecs,colors = ['#8cc4fc','#fcc48c','#fc8c8c']):
    for c,polys in zip(colors,allpolys):
        for poly in polys:
            if np.isnan(poly[0,0]): continue
            for i in range(0,len(poly),3):
                plt.fill([poly[i,0],poly[1+i,0],poly[2+i,0],poly[i,0]],[poly[i,1],poly[1+i,1],poly[2+i,1],poly[i,1]],facecolor=c,edgecolor=c)
    plt.grid()
    plt.axis('equal')
    for recs in allrecs:
        for rec in recs:
            plt.plot([rec[0,0],rec[1,0],rec[1,0],rec[0,0],rec[0,0]],[rec[0,1],rec[0,1],rec[1,1],rec[1,1],rec[0,1]],'k-')
    plt.xlabel('Easting / km')
    plt.ylabel('Northing / km')



def testing():
    #1x1 square has area 1.
    assert polytopeVolume(np.array([1,0,0,1,0,0,1,0,0,1,1,1]),2)==1
    
    #unit testing, a 2x2x2 cube centred at 0,0,0. Consisting of 12 tetrahedron.
    #we can replace this by one large cube.
    X = np.array([[1,1,-1,-1,-1,-1,-1,1,-1,0,0,0,1,1,-1,-1,-1,-1,1,-1,-1,0,0,0,  
                                 1,1,1,-1,-1,1,-1,1,1,0,0,0,1,1,1,-1,-1,1,1,-1,1,0,0,0,               
                                 1,-1,1,-1,-1,-1,-1,-1,1,0,0,0,1,-1,1,-1,-1,-1,1,-1,-1,0,0,0,  
                                 1,1,1,-1,1,-1,-1,1,1,0,0,0,1,1,1,-1,1,-1,1,1,-1,0,0,0,
                                 -1,1,1,-1,-1,-1,-1,-1,1,0,0,0,-1,1,1,-1,-1,-1,-1,1,-1,0,0,0,  
                                 1,1,1,1,-1,-1,1,-1,1,0,0,0,1,1,1,1,-1,-1,1,1,-1,0,0,0,
                                 ]])

    step = 0.2
    dims = 3
    polys = splitintopolygons(X[0,:],dims)
    allrecs = []
    for poly in polys:
        if np.isnan(poly[0,0]): continue
        vol = 1+int(5*np.sqrt(polytopeVolume(poly.flatten(),dims)))
        
        recs,ps,_ = compute_rectangles(poly,step,dims,Nrecs=1,Ntrials=50)
        allrecs.extend(recs)
        
    assert (polytopeVolume(X[0,:],3) == 8)
    ps = getinsidepoints(polys[0],.2,3)
    #assert(len(ps) == 1000)
    cuboidvol = np.prod(np.diff(allrecs[0],axis=0))
    assert np.abs(cuboidvol-8)<0.01, "Cuboid should have volume 8, but has volume %0.5f" % cuboidvol
