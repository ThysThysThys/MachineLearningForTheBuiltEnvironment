import matplotlib.pyplot as plt
import numpy as np

def plotter(xs,ys,zs):
    """
    Function to plot exisiting drone trajetory with calculated drone trajectory

    """
    plt.style.use('_mpl-gallery')

    x=np.array([2,1.08,-0.83,-1.97,-1.31,0.57])
    y=np.array([0,1.68,1.82,0.28,-1.51,-1.91])
    z=np.array([1,2.38,2.49,2.15,2.59,4.32])
    t=np.array([1,2,3,4,5,6])

    fig,ax=plt.subplots(subplot_kw={"projection":"3d"},figsize=(8,6))
    ax.set_xlabel(xlabel='X Axis',fontweight='bold')
    ax.set_ylabel(ylabel='Y Axis',fontweight='bold')
    ax.set_zlabel(zlabel='Z Axis',fontweight='bold')
    ax.scatter(x,y,z,color='red', s=50,label='Waypoints')
    ax.plot(x,y,z, color='blue',linewidth=2,label='Trajectory')
    ax.legend(loc='upper right')


    #Calculated points

    for xc,yc,zc in zip(xs,ys,zs):
        ax.scatter(xc,yc,zc,color='orange', s=50,label='Waypoints')
        ax.plot(xc,yc,zc, color='green',linewidth=2,label='Trajectory')
        #ax.legend(loc='upper right')

    for a,b,c,d in zip(x,y,z,t):
       ax.text(a+0.15,b+0.15,c+0.15, f"t={d}", fontsize=8,fontweight='bold')

    for xc,yc,zc in zip(xs,ys,zs):
        for a,b,c,d in zip(xc,yc,zc,t):
            ax.text(a+0.15,b+0.15,c+0.15, f"t={d}", fontsize=8,fontweight='bold')


    fig.suptitle("Drone Trajectory")
    plt.show()


if __name__ =='__main__':
    plotter()