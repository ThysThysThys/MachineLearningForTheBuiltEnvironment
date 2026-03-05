import matplotlib.pyplot as plt
import numpy as np

def sub_plotter(xs,ys,zs,ts):
    """
    Function to plot exisiting drone trajetory with calculated drone trajectory in 
    3 different axes plots

    Only for rough visualisation purposes

    """

    x=np.array([2,1.08,-0.83,-1.97,-1.31,0.57])
    y=np.array([0,1.68,1.82,0.28,-1.51,-1.91])
    z=np.array([1,2.38,2.49,2.15,2.59,4.32])
    t=np.array([1,2,3,4,5,6])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))

    # X-Y
    ax1.scatter(x, y, color='red', s=50, label='Measured')
    ax1.plot(x, y, color='blue', linewidth=2)
    ax1.scatter(xs, ys, color='orange', s=50, label='Fitted')
    ax1.plot(xs, ys, color='green', linewidth=2)

    """
    for a,b,ts in zip(x,y,t):
        ax1.text(a+0.15,b+0.15, f"t={ts}", fontsize=8,fontweight='bold')

    for a,b,k in zip(xs,ys,ts):
        ax1.text(a+0.15,b+0.15, f"t={k}", fontsize=8,fontweight='bold')

    """
    
    #ax1.set_xlabel('X')
    #ax1.set_ylabel('Y')
    ax1.set_title('X-Y Projection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Y-Z
    ax2.scatter(y, z, color='red', s=50, label='Measured')
    ax2.plot(y, z, color='blue', linewidth=2)
    ax2.scatter(ys, zs, color='orange', s=50, label='Fitted')
    ax2.plot(ys, zs, color='green', linewidth=2)

    """
    for a,b,ts in zip(y,z,t):
        ax2.text(a+0.15,b+0.15, f"t={ts}", fontsize=8,fontweight='bold')
    
    for a,b,ts in zip(ys,zs,t):
        ax2.text(a+0.15,b+0.15, f"t={ts}", fontsize=8,fontweight='bold')
    """ 

    #ax2.set_xlabel('Y')
    #ax2.set_ylabel('Z')
    ax2.set_title('Y-Z Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # X-Z
    ax3.scatter(x, z, color='red', s=50, label='Measured')
    ax3.plot(x, z, color='blue', linewidth=2)
    ax3.scatter(xs, zs, color='orange', s=50, label='Fitted')
    ax3.plot(xs, zs, color='green', linewidth=2)
    """
    for a,b,ts in zip(x,z,t):
        ax3.text(a+0.15,b+0.15, f"t={ts}", fontsize=8,fontweight='bold')

    for a,b,ts in zip(xs,zs,t):
        ax3.text(a+0.15,b+0.15, f"t={ts}", fontsize=8,fontweight='bold')
    """
    #ax3.set_xlabel('X')
    #ax3.set_ylabel('Z')
    ax3.set_title('X-Z Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    for ax in fig.get_axes():
        ax.label_outer()


#if __name__ =='__main__':
 #   sub_plotter()