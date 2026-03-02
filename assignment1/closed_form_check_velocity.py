import numpy as np


x=np.array([2,1.08,-0.83,-1.97,-1.31,0.57])
y=np.array([0,1.68,1.82,0.28,-1.51,-1.91])
z=np.array([1,2.38,2.49,2.15,2.59,4.32])
t=np.array([1,2,3,4,5,6])

#For Linear regression
def closed_form_check(axis,ts):


    t_mean=np.mean(ts)
    axis_mean=np.mean(axis)

    #print(t_mean)
    #print(axis_mean)

    covariance = np.mean((ts - t_mean) * (axis - axis_mean))
    #print(covariance)

    var=np.mean((ts - t_mean) ** 2)

    beta=(covariance/var)
    alpha=axis_mean - beta * t_mean

    print(f" Closed form Equation : \n axis = {alpha} {beta} * t")


if __name__ == '__main__':
    closed_form_check(x,t)
    closed_form_check(y,t)
    closed_form_check(z,t)

    
