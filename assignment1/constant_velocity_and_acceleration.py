import Plotter


# given data
pxs = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
pys = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
pzs = [1, 2.38, 2.49, 2.15, 2.59, 4.32]
ts = range(1, 7)

# running params
plot_seventh_point = True
do_constant_v = True
do_constant_a = True

# parameters
# constant v
iterations_constant_v = [1000]
learning_rate_constant_v = [0.01]
# constant a
iterations_constant_a = [10000]
learning_rate_constant_a = [0.0001]
tolerance = 0.0000001
initial_v = 1
initial_b = 0
initial_a = 0

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
        gradient += -2 * (p - (v * t + b))
    return gradient

# Perform gradient descent to determine best values for v and b
# Assumption that the function differs per axis, i.e. different v and b to be determined per axis
def gradient_descent_axis_constant(max_iterations, learning_rate, initial_v, initial_b, ps, ts):
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



def func_constant_acc(a, t, v, b):
    return a*(t**2) + v*t + b


def error_func_constant_acc(ps, ts, a, v, b):
    sse = 0
    for p, t in zip(ps, ts):
        sse += (p - func_constant_acc(a, t, v, b))**2
    return sse


def gradient_func_constant_acc_a(ps, ts, a, v, b):
    gradient = 0
    for p, t in zip(ps, ts):
        gradient += -2 * (t**2) * (p - (a*(t**2) + v*t + b))
    return gradient


def gradient_func_constant_acc_v(ps, ts, a, v, b):
    gradient = 0
    for p, t in zip(ps, ts):
        gradient += -2 * t * (p - (a*(t**2) + v*t + b))
    return gradient


def gradient_func_constant_acc_b(ps, ts, a, v, b):
    gradient = 0
    for p, t in zip(ps, ts):
        gradient += -2 * (p - (a*(t**2) + v*t + b))
    return gradient


def gradient_descent_axis_quadratic(max_iterations, learning_rate, initial_a, initial_v, initial_b, ps, ts):
    a = initial_a
    v = initial_v
    b = initial_b

    for i in range(max_iterations):
        ga = gradient_func_constant_acc_a(ps, ts, a, v, b)
        gv = gradient_func_constant_acc_v(ps, ts, a, v, b)
        gb = gradient_func_constant_acc_b(ps, ts, a, v, b)

        if (abs(ga * learning_rate) < tolerance and
            abs(gv * learning_rate) < tolerance and
            abs(gb * learning_rate) < tolerance):
            break

        a = a - learning_rate * ga
        v = v - learning_rate * gv
        b = b - learning_rate * gb

    return a, v, b

if __name__ == '__main__':
    if do_constant_v:
        # find best parameter values, strictly looking at the error
        best_error = 10000000
        best_xs=[]
        best_ys=[]
        best_zs=[]
        best_vs = []
        best_bs = []
        best_iterations = -1
        best_learning_rate = -1

        # gradient descent for constant velocity and plotting results
        for iterations in iterations_constant_v:
            for learning_rate in learning_rate_constant_v:
                # do gradient descent
                vx, bx = gradient_descent_axis_constant(iterations, learning_rate, initial_v, initial_b, pxs, ts)
                vy, by = gradient_descent_axis_constant(iterations, learning_rate, initial_v, initial_b, pys, ts)
                vz, bz = gradient_descent_axis_constant(iterations, learning_rate, initial_v, initial_b, pzs, ts)

                # calculate xyzs and determine SSE
                x=[]
                y=[]
                z=[]
                for i in ts:
                    x.append(func_constant_speed(vx,i,bx))
                    y.append(func_constant_speed(vy,i,by))
                    z.append(func_constant_speed(vz,i,bz)) 
                total_error = error_func_constant_speed(pxs, ts, vx, bx) + error_func_constant_speed(pys, ts, vy, by) + error_func_constant_speed(pzs, ts, vz, bz)
                
                if total_error < best_error:
                    best_error = total_error
                    best_xs = x
                    best_ys = y
                    best_zs = z
                    best_vs = [vx, vy, vz]
                    best_bs = [bx, by, bz]
                    best_iterations = iterations
                    best_learning_rate = learning_rate

        print(f"For constant velocity:\nBest error: {best_error}\nBest number of iterations: {iterations}\nBest learning rate: {learning_rate}\nV values per axis: {best_vs}\nB values per axis: {best_bs}\n")
        Plotter.plotter(best_xs, best_ys, best_zs, ts)

    if do_constant_a:
        # find best parameter values, strictly looking at the error
        best_error = 10000000
        best_xs=[]
        best_ys=[]
        best_zs=[]
        best_as = []
        best_vs = []
        best_bs = []
        best_iterations = -1
        best_learning_rate = -1

        # gradient descent for constant acceleration and plotting results
        for iterations in iterations_constant_a:
            for learning_rate in learning_rate_constant_a:
                # do gradient descent
                ax, vx, bx = gradient_descent_axis_quadratic(
                    iterations, learning_rate, initial_a, initial_v, initial_b, pxs, ts
                )
                ay, vy, by = gradient_descent_axis_quadratic(
                    iterations, learning_rate, initial_a, initial_v, initial_b, pys, ts
                )
                az, vz, bz = gradient_descent_axis_quadratic(
                    iterations, learning_rate, initial_a, initial_v, initial_b, pzs, ts
                )

                x = []
                y = []
                z = []
                for i in ts:
                    x.append(func_constant_acc(ax, i, vx, bx))
                    y.append(func_constant_acc(ay, i, vy, by))
                    z.append(func_constant_acc(az, i, vz, bz))
                total_error = error_func_constant_acc(pxs, ts, ax, vx, bx) + error_func_constant_acc(pys, ts, ay, vy, by) + error_func_constant_acc(pzs, ts, az, vz, bz)

                if plot_seventh_point:
                    t_next = 7

                    x7 = func_constant_acc(ax, t_next, vx, bx)
                    y7 = func_constant_acc(ay, t_next, vy, by)
                    z7 = func_constant_acc(az, t_next, vz, bz)

                    x.append(x7)
                    y.append(y7)
                    z.append(z7)

                if total_error < best_error:
                    best_error = total_error
                    best_xs = x
                    best_ys = y
                    best_zs = z
                    best_as = [ax, ay, az]
                    best_vs = [vx, vy, vz]
                    best_bs = [bx, by, bz]
                    best_iterations = iterations
                    best_learning_rate = learning_rate

        print(f"For constant acceleration:\nBest error: {best_error}\nBest number of iterations: {iterations}\nBest learning rate: {learning_rate}\nA values per axis: {best_as}\nV values per axis: {best_vs}\nB values per axis: {best_bs}")
        if plot_seventh_point:
            ts = range(1, 8)
        Plotter.plotter(best_xs, best_ys, best_zs, ts)
