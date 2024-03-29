# import the necessary packages
import os

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-6
MAX_LR = 1e-2
STEP_SIZE = 8
CLR_METHOD = "triangular2"

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["outputs", "lrfind_plot.pdf"])
TRAINING_PLOT_PATH = os.path.sep.join(["outputs", "training_plot.pdf"])
CLR_PLOT_PATH = os.path.sep.join(["outputs", "clr_plot.pdf"])
