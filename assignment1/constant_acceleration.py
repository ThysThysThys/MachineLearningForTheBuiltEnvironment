import numpy as np
import Plotter
import subplot_axis


x=np.array([2,1.08,-0.83,-1.97,-1.31,0.57])
y=np.array([0,1.68,1.82,0.28,-1.51,-1.91])
z=np.array([1,2.38,2.49,2.15,2.59,4.32])
t=np.array([1,2,3,4,5,6])
ts=np.array([1,2,3,4,5,6,7])


#For constant Acceleration
# y=w_0+w_1*t+w_2*t^2

#Learning rate
tolerance=1e-9
iterations=300000
learning_rate=0.00001




def SSE(w_0, w_1 ,w_2, y, t):
    """
    Function to calculate SSE (Sum of squared errors)
    Input:
        w_0 -> intercept
        w_1 -> coefficient of t
        w_2 -> coefficient of t^2
        y -> Actual measured points (1-Axis)
        t-> Time
    
    Output:
        Sum of squared error
    """

    total_error = 0

    for i,k in zip(y,t):
        y_pred = w_0 + (w_1 * k) + (w_2 * (k**2))
        total_error += (i - y_pred)**2

    return total_error


def gradient_descent(y,t,learning_rate,iterations):
    """
    Function to calculate gradient_descent 
    Input:
        y -> Original Points
        t-> Time
        learning_rate -> Learning rate
        iterations -> Number of iterations
        
    Output:
        Weight values
    """
    w_0,w_1,w_2 = 0.0, 0.0, 0.0

    #For checking tolerance
    prev_sse = float('inf')

    for i in range(iterations):


        residual= y - (w_0 + (w_1 * t) + (w_2 * (t**2)))

        grad_w_0 = -2 * np.sum(residual)
        grad_w_1 = -2 * np.sum(residual * t)
        grad_w_2 = -2 * np.sum(residual * t * t)

        #Updating the weights
        w_0 = w_0 - learning_rate * grad_w_0
        w_1 = w_1 - learning_rate * grad_w_1
        w_2 = w_2 - learning_rate * grad_w_2

        #Tolerance check for early exit of iterations
        curr_sse = SSE(w_0,w_1,w_2,y,t)
        if abs(prev_sse-curr_sse) < tolerance:
            print(f"Converged at Iteration : {i}")
            break
        
        prev_sse=curr_sse

    return w_0,w_1,w_2


#Calculating weights for respective axes
w0_x, w1_x, w2_x = gradient_descent(x,t,learning_rate,iterations)
w0_y, w1_y, w2_y = gradient_descent(y,t,learning_rate,iterations)
w0_z, w1_z, w2_z = gradient_descent(z,t,learning_rate,iterations)

print("Respective weights as per axes : ")
print(f"X Axis: w0={w0_x:.4f}, w1={w1_x:.4f}, w2={w2_x:.4f}")
print(f"Y Axis: w0={w0_y:.4f}, w1={w1_y:.4f}, w2={w2_y:.4f}")
print(f"Z Axis: w0={w0_z:.4f}, w1={w1_z:.4f}, w2={w2_z:.4f}")


#Calculating Sum of Squared errors

sse_x = SSE(w0_x, w1_x, w2_x, x, t)
sse_y = SSE(w0_y, w1_y, w2_y, y, t)
sse_z = SSE(w0_z, w1_z, w2_z, z, t)

print("\n Sum of Squared Errors:")
print(f"SSE X: {sse_x:.4f}")
print(f"SSE Y: {sse_y:.4f}")
print(f"SSE Z: {sse_z:.4f}")

total_sse=sse_x+sse_y+sse_z
print(f"Total SSE: {total_sse:.4f}")


xs=[]
ys=[]
zs=[]


for i in t:
    temp_x = w0_x + w1_x * i + w2_x * pow(i,2)
    temp_y = w0_y + w1_y * i + w2_y * pow(i,2)
    temp_z = w0_z + w1_z * i + w2_z * pow(i,2)

    xs.append(temp_x)
    ys.append(temp_y)
    zs.append(temp_z)

#print(xs,"\n",ys,"\n",zs,"\n")

#Calculation for t=7 seconds

temp_x = w0_x + w1_x * 7 + w2_x * pow(7,2)
temp_y = w0_y + w1_y * 7 + w2_y * pow(7,2)
temp_z = w0_z + w1_z * 7 + w2_z * pow(7,2)

print(f"\n Trajectory Co-ordinates at t = 7 : \n X = {temp_x}\n Y = {temp_y} \n Z = {temp_z}")

xs.append(temp_x)
ys.append(temp_y)
zs.append(temp_z)

Plotter.plotter(xs,ys,zs,ts)
subplot_axis.sub_plotter(xs,ys,zs,ts)






    


