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

#round (by probability)
#https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return int(sign * round_func(x))
    
def sharededge(pgb1,pgb2):
    #TODO Should check these are two contiguous vertices
    return np.sum([np.any(np.all(pgb1==vert,1)) for vert in pgb2])>1 #two points that match = edge that matches


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

        
        
def polytopeVolume(polytope,input_space_dim,spheres=False):
    """
    Pass a 1d array as a list of simplexes, e.g. a square made of two triangles:
    [1,0,0,1,0,0, 1,0,0,1,1,1]
    """
    if spheres:
        from scipy.special import gamma
        return np.sum([(np.pi**(input_space_dim/2)/gamma((input_space_dim/2) + 1))*(r**input_space_dim) for r in polytope[input_space_dim::((input_space_dim+1)*input_space_dim)]])
    
        
    Ncoords = input_space_dim*(input_space_dim+1)
    vol = 0
    for i in range(0,len(polytope),Ncoords):
        vectors = polytope[i:(i+Ncoords)].reshape(input_space_dim+1,input_space_dim)
        if np.isnan(vectors[0,0]):
            break
        vol += simplexVolume(vectors,input_space_dim)
    return vol
    

def pointsinrec(ps,rec):
    return np.all((ps>rec[::2]) & (ps<rec[1::2]),1)

def getinsidepoints(pgb,step,dims):
    testpoints = None
    if np.isnan(pgb[0,0]):
        return np.array([])
    for start,end in zip(np.min(pgb,0),np.max(pgb,0)):
        testpoints_1d = np.arange(start,end,step)
        if testpoints is None:
            testpoints = testpoints_1d[:,None]
        else:
            testpoints=(np.c_[np.tile(testpoints,[len(testpoints_1d),1]),testpoints_1d.repeat(len(testpoints))[:,None]])
    insidepoints = np.zeros([0,testpoints.shape[1]])
    area = 0
    #testpoints += np.random.rand(testpoints.shape[0],testpoints.shape[1])*0.01
    for i in range(0,len(pgb),dims+1):
        try:        
            hull = Delaunay(pgb[i:i+dims+1,:])
            insidepoints=np.r_[insidepoints,(testpoints[hull.find_simplex(testpoints)>=0,:])]
        except QhullError:
            pass #skip - the triangle's probably just flat
    return np.array(insidepoints)
    
def getinsidepoints_sphere(pgb,step,dims):
    testpoints = None
    if np.isnan(pgb[0,0]):
        return np.array([])
    for start,end in zip(pgb[0,:]-pgb[1,0],pgb[0,:]+pgb[1,0]):
        testpoints_1d = np.arange(start,end,step)
        if testpoints is None:
            testpoints = testpoints_1d[:,None]
        else:
            testpoints=(np.c_[np.tile(testpoints,[len(testpoints_1d),1]),testpoints_1d.repeat(len(testpoints))[:,None]])
    #insidepoints = np.zeros([0,testpoints.shape[1]])
    insidepoints=[]
    for p in testpoints:
        if np.sum((p-pgb[0,:])**2)<(pgb[1,0])**2:
            insidepoints.append(p)

    return np.array(insidepoints)    

def compute_rectangles(polygon,step,dims,Nrecs=10,Ntrials=10,failcountlimit=9,spheres=False):
    """
    Given a 2d (NxD) array, consisting of vertex locations on each row, and each d+1 rows containing one simplex
    Computes a list of rectangles. Each element of the output is a 2xd array, with the top row being one
    corner of the rectangle, and the bottom the opposite corner.
    
    Nrecs = number of rectangles to use
    Ntrials = number of times we'll try to get the best rectangle (greedy)
    failcountlimit = in the inner loop we try different directions until we make progress. If we fail 5 times we give up.
    """
    if spheres:
        ps = getinsidepoints_sphere(polygon,step,dims)
    else:
        ps = getinsidepoints(polygon,step,dims)
    initiallen = len(ps)
    ps = np.unique(ps,axis=0)
    if len(ps)!=initiallen:
        print("WARNING: You have overlapping triangles.")
    ps+=np.random.rand(ps.shape[0],ps.shape[1])*0.0000001
    oldps = ps.copy()
    recs = []
    for it in range(Nrecs):
        if len(ps)==0: #no more points to cover.
            break
        #print(".",end='')
        bestrec = None
        bestsize = -1
        for it in range(Ntrials):
            #TODO: Split into polygons again so the median makes more sense!
            found = np.where(np.all(ps == np.median(ps,0),1))[0]
            if len(found)==0: #issue if the shape is so convex the median point isn't in the shape
                found = np.random.randint(0,len(ps))
            else:
                found = found[0]
            rec = np.array(ps[found,:])[None,:].repeat(2,0)
            rec[0,:]-=step/2
            rec[1,:]+=step/2
            recsize = np.ones(rec.shape[1])
            #recs = [rec]
            #try expanding rectangle in dim
            failcount = 0
            while failcount<failcountlimit:
                dim = np.random.randint(0,rec.shape[1])
                before = np.sum(pointsinrec(ps,rec))
                trialrec = rec.copy()
                changedir = np.random.randint(0,2)
                if changedir==1: 
                    change = step
                else:
                    change = -step
                trialrec[changedir,dim]+= change

                after = np.sum(pointsinrec(ps,trialrec))
                expectedincrease = np.prod(recsize)/recsize[dim]
                if after - before != expectedincrease:
                    failcount+=1
                    #do something else:
                    continue
                failcount = 0
                rec = trialrec
                recsize[dim]+=1
            if np.prod(recsize)>bestsize:
                bestsize = np.prod(recsize)
                bestrec = rec
        ps = ps[~pointsinrec(ps,bestrec)]
        recs.append(bestrec)
    return recs,ps,oldps


def gettrainingdata(Xrow,dims,Nrecs,step,Ntrials=50,failcountlimit=40,spheres=False):
    """
    Pass one row of X (a list of simplexes, e.g. 0,1,1,0,1,1,0,1,1,0,0,0,2,3,3,2,3,3,2,3,3,2,2,2 = two squares)
    Returns a bunch of rectangles that fills them all.
    """
    #compute the sum of the squareroots of the volumes of the polytopes.
    
    
    polys = splitintopolygons(Xrow,dims)
    
    npolys = 0
    vols = np.zeros(len(polys))    
    for i,poly in enumerate(polys):
        if np.isnan(poly[0,0]): continue
        #vols[i] = np.sqrt(polytopeVolume(poly.flatten(),dims))
        vols[i] = (polytopeVolume(poly.flatten(),dims,spheres))
        npolys+=1
    
    allrecs = []
    volcorrections = []
    for v,poly in zip(vols,polys):
        if np.isnan(poly[0,0]): continue
        #we give each polygon one rectangle, so that leaves Nrecs-npolys rectangles to play with
        #so each polygon gets 1+(Nrec-npolys)*(vol/sum of vol)
        target = prob_round(1+(Nrecs-npolys)*v/np.sum(vols))
        #target = prob_round(1+(Nrecs-npolys)*np.sqrt(v)/np.sum(np.sqrt(vols)))
        #print(1+(Nrecs-npolys)*v/np.sum(vols))
        recs,ps,oldps = compute_rectangles(poly,step,2,Nrecs=target,Ntrials=Ntrials,failcountlimit=failcountlimit,spheres=spheres)
        volcorrections.extend([v/getvolumes(recs)]*len(recs))
        allrecs.extend(recs)
    return allrecs, polys, np.sum(vols), volcorrections, ps, oldps

def compute_newX(X,Nrecs=10,step=0.025,Ntrials=10,failcountlimit=7,dims=2,spheres=False):
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
    allvolcorrections = []
    for Xrow in X:
        recs,polys,vol,volcorrections, ps, oldps = gettrainingdata(Xrow,dims,Nrecs=Nrecs,step=step,Ntrials=Ntrials,failcountlimit=failcountlimit,spheres=spheres)
        newX.append(np.array([rec.T.flatten() for rec in recs]))
        allpolys.append(polys)
        allrecs.append(recs)
        allvols.append(vol)
        allps.append(ps)
        alloldps.append(oldps)
        allvolcorrections.append(volcorrections)

    startindices = np.r_[np.array([0]),np.cumsum([len(n) for n in newX])]
    approxvols = [getvolumes(recs) for recs in allrecs]
    allvolscales = np.array(allvols)/np.array(approxvols)        
    return newX, startindices,allvolscales,allvolcorrections, allpolys, allrecs



def getvolumes(recs):
    return np.sum([np.prod(np.abs(rec[1::2]-rec[0::2])) for rec in recs])

def plotpolys(allpolys,allrecs,colors = ['#8cc4fc','#fcc48c','#fc8c8c'],spheres=False):
    for c,polys in zip(colors,allpolys):
        for poly in polys:
            if np.isnan(poly[0,0]): continue
            if not spheres:
                for i in range(0,len(poly),3):
                    plt.fill([poly[i,0],poly[1+i,0],poly[2+i,0],poly[i,0]],[poly[i,1],poly[1+i,1],poly[2+i,1],poly[i,1]],facecolor=c,edgecolor=c)
            else:
                ax = plt.gca()
                circle = plt.Circle(poly[0,:], poly[1,0], color=c)
                ax.add_artist(circle)
    plt.grid()
    plt.axis('equal')
    for recs in allrecs:
        for rec in recs:
            plt.plot([rec[0,0],rec[1,0],rec[1,0],rec[0,0],rec[0,0]],[rec[0,1],rec[0,1],rec[1,1],rec[1,1],rec[0,1]],'k-')
    plt.xlabel('Easting / km')
    plt.ylabel('Northing / km')



def testing():
    #spheres!
    assert polytopeVolume(np.array([1,2]),1,spheres=True)==2*2
    assert polytopeVolume(np.array([1,0,2]),2,spheres=True)==np.pi*2**2
    assert np.abs(polytopeVolume(np.array([0,1,0,2]),3,spheres=True)-(4/3)*np.pi*2**3)<1e-10
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
        
        recs,ps,_ = compute_rectangles(poly,step,dims,Nrecs=1,Ntrials=50,failcountlimit=40)
        allrecs.extend(recs)
        
    assert (polytopeVolume(X[0,:],3) == 8)
    ps = getinsidepoints(polys[0],.2,3)
    #assert(len(ps) == 1000)
    cuboidvol = np.prod(np.diff(allrecs[0],axis=0))
    assert np.abs(cuboidvol-8)<0.01, "Cuboid should have volume 8, but has volume %0.5f" % cuboidvol
