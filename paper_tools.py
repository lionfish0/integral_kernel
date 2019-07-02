import census
import GPy
#from shapeintegrals_fast import ShapeIntegral
from shapeintegrals_fast import ShapeIntegral
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
from shapely import geometry

def compute_centroids_and_areas(X):
    centroids = []
    areas = []
    for x in X:
        polys = []
        for t in x.reshape(int(len(x)/6),6):
            if np.isnan(t[0]):
                break
            polys.append(geometry.Polygon([[pxs,pys] for pxs,pys in zip(t[::2],t[1::2])]))
        mp = geometry.MultiPolygon(polys)
        centroids.append([mp.centroid.x,mp.centroid.y])
        areas.append(mp.area)
    return np.array(centroids), np.array(areas)[:,None]

def plotmap(X,Y):
    """
    Pass function, X: list of triangles in standard X format
    and Y: population for each row of X.
    
    Plots a map of population densities
    """
    _, vols = compute_centroids_and_areas(X)
    
    fig, ax = plt.subplots(figsize=[10,10])
    patches = []
    colors = []
    for x,pop,vol in zip(X,Y,vols[:,0]):
        for triangle in x.reshape([int(len(x)/6),6]):
            if np.isnan(triangle[0]):
                break
            polygon = Polygon(triangle.reshape(3,2), True)
            patches.append(polygon)
            colors.append((pop/vol)[0])
        if vol>1:
            colors[-1] = np.random.rand()*100
    #p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    p = PatchCollection(patches, cmap='gray', alpha=1.0)
    #colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))
    ax.add_collection(p)
    minx = np.nanmin(X[:,::2])
    maxx = np.nanmax(X[:,::2])
    miny = np.nanmin(X[:,1::2])
    maxy = np.nanmax(X[:,1::2])  
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.grid()
    plt.axis('equal')
    plt.xlabel('Easting / km')
    plt.ylabel('Northing / km')
    plt.show()
def normalise_area(X,Y):
    """
    Normalise the slightly more awkward area inputs.
    We need to subtract the mean population density.
    """
    
    _, areas = compute_centroids_and_areas(X)
    avg_den = np.sum(Y) / np.sum(areas[:,0])
    Y = Y - avg_den * areas
    return Y, avg_den

def unnormalise_area(X,Y,avg_den):
    _, areas = compute_centroids_and_areas(X)
    Y = Y + avg_den * areas
    return Y
    

def build_integral_model(X,Y,testX,Nperunit=20, lengthscale=0.16):
    """Builds integral model"""
    kern = GPy.kern.Matern32(2,variance=50.0, lengthscale=1.0,ARD=False)
    k = ShapeIntegral(X.shape[1],Nperunit=Nperunit,kernel=kern)
    m = GPy.models.GPRegression(X,Y,k,normalizer=False)
    m.Gaussian_noise = 1
    #m.kern.lengthscale.fix(lengthscale)
    m.kern.kernel.lengthscale.fix(lengthscale)
    return m

def build_simple_model(X,Y,testX, lengthscale=0.16):
    centroidX, areas = compute_centroids_and_areas(X)
    k = GPy.kern.Matern32(2,variance=50.0, lengthscale=1.0,ARD=False)

    m = GPy.models.GPRegression(centroidX,Y/areas,k,normalizer=False)#TODO Can Turn ON normaliser for the simple model!!!
    m.Gaussian_noise = 1
    m.kern.lengthscale.fix(lengthscale)
    return m

def print_predictions(pred, predvar, testY, testX, N=5):
    centroidtestX, test_areas = compute_centroids_and_areas(testX)
    print("Population Predictions")
    for i,(p,pv,act,testx,vol) in enumerate(zip(pred,predvar,testY,testX,test_areas)):
        print("%7.0f +/- %7.0f (act: %7.0f) [vol=%0.3f]"%(p,np.sqrt(pv),act,vol))
        if (i>N): #just show first N
            break

    print("Population Density Predictions")
    for i,(p,pv,act,testx,vol) in enumerate(zip(pred,predvar,testY,testX,test_areas)):
        print("%7.0f +/- %7.0f (act: %7.0f)"%(p/vol,np.sqrt(pv)/vol,act/vol))
        if (i>N): #just show first N
            break
            
def get_rmse(testX,testY,preds,threshold=np.infty):
    centroidtestX, test_areas = compute_centroids_and_areas(testX)
    tempkeep = np.where((testY/test_areas)<threshold)
    return np.sqrt(np.mean(((np.array(testY)/test_areas)-(preds[:,0]/test_areas))[tempkeep,:]**2))

def get_meanabserror(testX,testY,preds,threshold=np.infty):
    centroidtestX, test_areas = compute_centroids_and_areas(testX)
    tempkeep = np.where((testY/test_areas)<threshold)
    return np.mean(np.abs(((np.array(testY)/test_areas)-(preds[:,0]/test_areas))[tempkeep,:]))

    
def plot_results_comparison(testX,testY,preds,size=2,marker='.',color='k',label=None): #todo finish coding
    #plot to compare densities
    centroidtestX, test_areas = compute_centroids_and_areas(testX)
    plt.scatter(np.array(testY)/test_areas,preds/test_areas,size,marker=marker,color=color,label=label)
    plt.plot([0,25000],[0,25000],'k-')
    plt.grid()
    plt.xlabel('Actual (people/km$^2$)')
    plt.ylabel('Predicted (people/km$^2$)')
    plt.title('Population Density')
    plt.xlim([0,25000])
    
#TODO The following two functions basically do the same thing but for the two
#models, need to combine them into one function.

def plot_results_space_simple(X, Y, testX, testY, bbox, m):
    centroidtestX, test_areas = compute_centroids_and_areas(testX)
    centroidX, areas = compute_centroids_and_areas(X)    
    
    for x, y, vol in zip(centroidX, Y, areas):
        den = y/(vol)
        plt.plot(np.nanmean(x[1::2]),den,'xb')  

    for x, y in zip(centroidtestX, testY/test_areas):
        den = y        
        plt.plot(np.nanmean(x[1::2]),den,'.k',markersize=1)  
        
    #for xpos,col in zip(np.linspace(328,335,10),np.linspace(0,0.8,10)):
    for xpos,col in zip(np.linspace(bbox[0],bbox[1],10),np.linspace(0,0.8,10)):        
        positions = []

        for ypos in np.arange(bbox[2],bbox[3],0.1):
            positions.append([xpos,ypos])
        positions = np.array(positions)
        preds,_ = m.predict(positions)#census.getpred(m,positions)
        plt.plot(positions[:,1],preds,str(col))

    plt.xlabel('Northing')
    plt.ylabel('Population density (people/km$^2$)')  

def plot_results_space_integral(X, Y, testX, testY,bbox,m):
    vols = []

    for x, y in zip(X, Y):
        vol = m.kern.polytopeVolume(x)
        den = y/(vol)
        plt.plot(np.nanmean(x[1::2]),den,'xb')  
        
    for x, y in zip(testX, testY):
        vol = m.kern.polytopeVolume(x)
        den = y/vol
        plt.plot(np.nanmean(x[1::2]),den,'.k',markersize=1)  
        
    for xpos,col in zip(np.linspace(bbox[0],bbox[1],10),np.linspace(0,0.8,10)):    
        positions = []
        for ypos in np.arange(bbox[2],bbox[3],0.1):
            positions.append([xpos,ypos])
        positions = np.array(positions)
        preds = census.getpred(m,positions)
        plt.plot(positions[:,1],1e6*preds,str(col))

    plt.xlabel('Northing')
    plt.ylabel('Population density (people/km$^2$)')
