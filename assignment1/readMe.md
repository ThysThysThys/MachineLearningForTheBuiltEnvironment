# Instructions
This readMe will go over dependencies and how to run the code to reproduce the results of the gradient descent method for both the constant velocity and constant acceleration models.

## Dependencies
- Numpy
- MatPlotLib

## How to run the code?
There are two variants of the gradient descent model, namely the constant velocity variant and the constant acceleration variant. For the latter it was separately implemented by two teammembers, hence there are two versions.

**Constant velocity**: The constant velocity variant can be run by calling the main function of the *constant_velocity_and_acceleration.py* file. The gradient descent will be done and the plot will pop up automatically. Various parameters such as *learning rate* and *number of iterations* can be set at the top of the file. Multiple values for those two in particular can be set in order to determine the values that lead to the lowest error. Cartesian product combinations of all values in the lists will be tested.

**Constant acceleration**: The constant acceleration variant can be run by calling the main function of the *constant_acceleration.py* file. For this file, too, the gradient descent will be run and a plot will pop up automatically. Various parameters such as *learning rate* and *number of iterations* can be set at the top of the file. It can also be run via the *constant_velocity_and_acceleration.py* file, just like the constant variant.

## Other files
- Plotter.py and subplot_axis.py are used for plotting the results. plotter_old.py is an outdated initial version of it.
- closed_form_check_velocity.py is a way to calculate the closed form solution of the linear regression problem. It can be used to compare the outcome of the gradient descent.