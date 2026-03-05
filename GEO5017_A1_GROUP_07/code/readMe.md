# Instructions
This readMe will go over dependencies and how to run the code to reproduce the results of the gradient descent method for both the constant velocity and constant acceleration models.

## Dependencies
- Numpy
- MatPlotLib

## How to run the code?
**Constant velocity**: The constant velocity variant can be run by calling the main function of the *main.py* file, whilst the *do_constant_v* flag is set to *True*. The gradient descent will be done and the plot will pop up automatically. Various parameters such as *learning rate* and *number of iterations* can be set at the top of the file. Multiple values for those two in particular can be set in order to determine the values that lead to the lowest error. Cartesian product combinations of all values in the lists will be tested.

**Constant acceleration**: The constant acceleration variant can also be run by calling the main function of the *main.py* file, this time whilst setting the *do_constant_a* flag to *True*. There is also a flag for plotting the seventh point for the acceleration model. For this model, too, the gradient descent will be run and a plot will pop up automatically. Various parameters such as *learning rate* and *number of iterations* can be set at the top of the file.

## Other files
- Plotter.py and subplot_axis.py are used for plotting the results.
- closed_form_check_velocity.py is a way to calculate the closed form solution of the linear regression problem. It can be used to compare the outcome of the gradient descent of the constant velocity model.