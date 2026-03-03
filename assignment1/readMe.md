# Instructions
This readMe will go over dependencies and how to run the code to reproduce the results of the gradient descent method for both the constant velocity and constant acceleration models.

## Dependencies
- Numpy
- MatPlotLib

## How to run the code?
**Constant velocity**: The constant velocity variant can be run by calling the main function of the *constant_velocity.py* file. The gradient descent will be done and the plot will pop up automatically. Various parameters such as *learning rate* and *number of iterations* can be set at the top of the file.

**Constant acceleration**: The constant acceleration variant can be run by calling the main function of the *constant_acceleration.py* file. For this file, too, the gradient descent will be run and a plot will pop up automatically. Various parameters such as *learning rate* and *number of iterations* can be set at the top of the file.

## Other files
- Plotter.py and subplot_axis.py are used for plotting the results. plotter_old.py is an outdated initial version of it.
- closed_form_check_velocity.py is a way to calculate the closed form solution of the linear regression problem. It can be used to compare the outcome of the gradient descent.