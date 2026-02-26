import Plotter


# given data
pxs = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
pys = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
pzs = [1, 2.38, 2.49, 2.15, 2.59, 4.32]
ts = range(1, 7)


# parameters
max_iterations = 100
learning_rate = 0.01
tolerance = 0.001
initial_v = 1
initial_b = 0

"""
Constant speed function model: p = v * t + b
With:
    p = position
    v = velocity
    t = time
    b = initial position
p and t are given, while v and b need to be determined using gradient descent polynomial regression
"""
# Function constant speed
def func_constant_speed(v, t, b):
    return v * t + b

# Error function constant speed
def error_func_constant_speed(ps, ts, v, b):
    """
    ps -> True Position  (Given values)
    ts -> True Time (Given value)
    v -> Estimated Speed/Velocity
    b -> Estimated Starting Position
    """
    sse = 0
    for p, t in zip(ps, ts):
        error = (p - func_constant_speed(v, t, b))**2
        sse += error
    return sse

# Gradient for constant speed function wrt v
def gradient_func_constant_speed_v(ps, ts, v, b):
    """
    ps -> True Position  (Given values)
    ts -> True Time (Given value)
    v -> Estimated Speed/Velocity
    b -> Estimated Starting Position
    """
    gradient = 0
    for p, t in zip(ps, ts):
        #print(p, t)
        gradient += -2 * t * (p - (v * t + b))
    return gradient

# Gradient for constant speed function wrt b
def gradient_func_constant_speed_b(ps, ts, v, b):
    gradient = 0
    for p, t in zip(ps, ts):
        gradient += 2 * (p - (v * t + b))
    return gradient

# Perform gradient descent to determine best values for v and b
# Assumption that the function differs per axis, i.e. different v and b to be determined per axis
# TODO: make modular via list
def gradient_descent_axis(max_iterations, learning_rate, initial_v, initial_b, ps, ts):
    v = initial_v
    b = initial_b
    for i in range(max_iterations):
        gradient_v = gradient_func_constant_speed_v(ps, ts, v, b)
        gradient_b = gradient_func_constant_speed_b(ps, ts, v, b)
        if abs(gradient_b * learning_rate) < tolerance and abs(gradient_v * learning_rate) < tolerance:
            break
        if abs(gradient_v * learning_rate) > tolerance:
            v = v - learning_rate * gradient_v
        if abs(gradient_b * learning_rate) > tolerance:
            b = b - learning_rate * gradient_b
    return v, b

# test gradient descent for certain axis

xs=[]
ys=[]
zs=[]

for iterations in (10,25,100):
    for learning_rate in (0.01,0.01,0.001):
        for initial_v in (1,2,3):
            vx, bx = gradient_descent_axis(iterations, learning_rate, initial_v, initial_b, pxs, ts)
            vy, by = gradient_descent_axis(iterations, learning_rate, initial_v, initial_b, pys, ts)
            vz, bz = gradient_descent_axis(iterations, learning_rate, initial_v, initial_b, pzs, ts)

            x=[]
            y=[]
            z=[]

            for i in ts:
                x.append(func_constant_speed(vx,i,bx))
                y.append(func_constant_speed(vy,i,by))
                z.append(func_constant_speed(vz,i,bz)) 
            xs.append(x)
            ys.append(y)
            zs.append(z)
#print("v: ", v)
#print("b: ", b)
#print("Error: ", error_func_constant_speed(pzs, ts, v, b))


Plotter.plotter(xs,ys,zs)

