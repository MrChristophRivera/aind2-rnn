import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import keras
from string import ascii_lowercase as letters, digits

def window_transform_series(series, window_size=2): 
    """Windows an input series to create input output pairs
    eg if series is np.array([1,2,3,4]) and window size is 2ZeroDivisionError
    then returns x = np.array([[1,2],[2,3]]) and y = [ 3, 4]
    
    Args: 
        series(np.array): The input array
        window_size(int): the window size
    Returns: 
        X(np.array)
        y(np.array)
    """
    
    l = len(series)
    steps = l - window_size
    X = [series[i:i+window_size] for i in range(steps)] 
    y = [series[i+window_size] for i in range(steps)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def build_part1_RNN(window_size):
    """ Builds the first part of an RNN 
    Args: 
        window_size(int): the size of the window
    Returns: 
        model(keras.Sequential): The rnn model
    """
    
    model = Sequential()
    model.add(LSTM(units = 5, input_shape =(window_size, 1)))
    model.add(Dense(units = 1))
    return model


def cleaned_text(text):
    """ Cleans text by converting to lowercase, and replacing all non wanted characters with blanks
    Args: 
        text(str): the text
    
    Returns: 
        text(str)
    """
    punctuation = ['!', ',', '.', ':', ';', '?']
    allowed = list(letters) + punctuation
    text = text.lower()
    
    not_allowed = [char for char in set(text) if char not in allowed]
    for char in not_allowed: 
        text = text.replace(char, ' ')
    return text


def window_transform_text(text, window_size=2, step_size=1):
    """ Windows text into input output pairs
    Args: 
        text(str): the text 
        window_size(int): the size of the window
        step_size(int): the distance between step_size
    Returns: 
        inputs(list): list of windows
        output(list): list of outputs
    """
    l = len(text)
    steps = np.arange(start=-0, stop=l-window_size, step = step_size)
    
    inputs = [text[i:i+window_size] for i in steps]
    outputs = [text[step+window_size] for step in steps]

    return inputs,outputs


def build_part2_RNN(window_size, num_chars):
    """ Constructs a RNN model for character to character prediction. 
    Args: 
        window_size(int): the size of the input window_size
        num_chars(int): the number of characters in the vocab. 
    Returns: 
        model(keras.Sequential): The model 
    """
    model = Sequential()
    model.add(LSTM(units = 200, input_shape = (window_size, num_chars)))
    model.add(Dense(units = num_chars, activation =None))
    model.add(Activation('softmax'))
    return model
    