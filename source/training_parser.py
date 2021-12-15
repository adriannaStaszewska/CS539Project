import matplotlib.pyplot as plt
import re
import sys

# Additional documentation: I simply used the console to concatenate
# files with cat

# Searches through the data and finds instances of str <number>
def getNumbers(data, str):
    # Make a regular expression
    pattern = re.compile('(' + str + ')(' + r'\s*\d*[.]?\d*\s*' + ')', re.DOTALL)
    # print(pattern)
    # Return the numbers found
    return [float(i[2]) for i in re.finditer(pattern, data)]

# Helper function that makes a list of epochs
def getEpochs(y):
    return list(range(len(y)))

# Draws plots
def makePlot(loss, val_loss):
    plt.plot(getEpochs(loss), loss) # Plot loss
    plt.plot(getEpochs(val_loss), val_loss) # Plot validation loss
    plt.legend() # Make a legend
    plt.show() # Show the plot

# Main function
def main(data): # Pass file data in
    loss = getNumbers(data, r'- loss:') # Gets the losses
    val_loss = getNumbers(data, r'- val_loss:') # Gets the validation losses
    makePlot(loss, val_loss) # Makes a plot

# Handle arguments if ran directly
if __name__ == '__main__': # Check if this file was ran directly
    encoding = 'UTF-16' # Constant to be used later
    if len(sys.argv) >= 2: # Check for necessary arguments
        with open(sys.argv[1], 'rb') as in_file: # Open the file
            main(in_file.read().decode(encoding)) # Why was this file automatically in utf-16?