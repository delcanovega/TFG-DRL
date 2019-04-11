import numpy as np
import matplotlib.pyplot as plt
import sys

# Usage:
# launch from command line: python results_tool.py <input_file_name>
input_file = sys.argv[1]

# load data from input_file into a variable
data = np.loadtxt(input_file)

# Time to work with data...
