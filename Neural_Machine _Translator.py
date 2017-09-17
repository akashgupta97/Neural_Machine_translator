from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
%matplotlib inline


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

dataset[:10]

'''
[('9 may 1998', '1998-05-09'),
 ('10.09.70', '1970-09-10'),
 ('4/28/90', '1990-04-28'),
 ('thursday january 26 1995', '1995-01-26'),
 ('monday march 7 1983', '1983-03-07'),
 ('sunday may 22 1988', '1988-05-22'),
 ('tuesday july 8 2008', '2008-07-08'),
 ('08 sep 1999', '1999-09-08'),
 ('1 jan 1981', '1981-01-01'),
 ('monday may 22 1995', '1995-05-22')]
 
'''


Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

'''

X.shape: (10000, 30)
Y.shape: (10000, 10)
Xoh.shape: (10000, 30, 37)
Yoh.shape: (10000, 10, 11)
'''

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])

'''

Source date: 9 may 1998
Target date: 1998-05-09

Source after preprocessing (indices): [12  0 24 13 34  0  4 12 12 11 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36
 36 36 36 36 36]
Target after preprocessing (indices): [ 2 10 10  9  0  1  6  0  1 10]

Source after preprocessing (one-hot): [[ 0.  0.  0. ...,  0.  0.  0.]
 [ 1.  0.  0. ...,  0.  0.  0.]
 [ 0.  0.  0. ...,  0.  0.  0.]
 ..., 
 [ 0.  0.  0. ...,  0.  0.  1.]
 [ 0.  0.  0. ...,  0.  0.  1.]
 [ 0.  0.  0. ...,  0.  0.  1.]]
Target after preprocessing (one-hot): [[ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
 [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]
'''

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


# GRADED FUNCTION: one_step_attention

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor to propagate concat through a small fully-connected neural network to compute the "energies" variable e. (≈1 lines)
    e = densor(concat)
    # Use activator and e to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(e)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###

    return context



n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)



# GRADED FUNCTION: model# GRADED

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model([X, s0, c0], outputs)

    ### END CODE HERE ###

    return model



model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

model.summary()


'''
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_2 (InputLayer)             (None, 30, 37)        0                                            
____________________________________________________________________________________________________
s0 (InputLayer)                  (None, 128)           0                                            
____________________________________________________________________________________________________
bidirectional_1 (Bidirectional)  (None, 30, 128)       52224       input_2[0][0]                    
____________________________________________________________________________________________________
repeat_vector_1 (RepeatVector)   (None, 30, 128)       0           s0[0][0]                         
                                                                   lstm_1[0][0]                     
                                                                   lstm_1[1][0]                     
                                                                   lstm_1[2][0]                     
                                                                   lstm_1[3][0]                     
                                                                   lstm_1[4][0]                     
                                                                   lstm_1[5][0]                     
                                                                   lstm_1[6][0]                     
                                                                   lstm_1[7][0]                     
                                                                   lstm_1[8][0]                     
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 30, 256)       0           bidirectional_1[0][0]            
                                                                   repeat_vector_1[0][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[1][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[2][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[3][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[4][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[5][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[6][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[7][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[8][0]            
                                                                   bidirectional_1[0][0]            
                                                                   repeat_vector_1[9][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 30, 1)         257         concatenate_1[0][0]              
                                                                   concatenate_1[1][0]              
                                                                   concatenate_1[2][0]              
                                                                   concatenate_1[3][0]              
                                                                   concatenate_1[4][0]              
                                                                   concatenate_1[5][0]              
                                                                   concatenate_1[6][0]              
                                                                   concatenate_1[7][0]              
                                                                   concatenate_1[8][0]              
                                                                   concatenate_1[9][0]              
____________________________________________________________________________________________________
attention_weights (Activation)   (None, 30, 1)         0           dense_1[0][0]                    
                                                                   dense_1[1][0]                    
                                                                   dense_1[2][0]                    
                                                                   dense_1[3][0]                    
                                                                   dense_1[4][0]                    
                                                                   dense_1[5][0]                    
                                                                   dense_1[6][0]                    
                                                                   dense_1[7][0]                    
                                                                   dense_1[8][0]                    
                                                                   dense_1[9][0]                    
____________________________________________________________________________________________________
dot_1 (Dot)                      (None, 1, 128)        0           attention_weights[0][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[1][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[2][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[3][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[4][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[5][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[6][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[7][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[8][0]          
                                                                   bidirectional_1[0][0]            
                                                                   attention_weights[9][0]          
                                                                   bidirectional_1[0][0]            
____________________________________________________________________________________________________
c0 (InputLayer)                  (None, 128)           0                                            
____________________________________________________________________________________________________
lstm_1 (LSTM)                    [(None, 128), (None,  131584      dot_1[0][0]                      
                                                                   s0[0][0]                         
                                                                   c0[0][0]                         
                                                                   dot_1[1][0]                      
                                                                   lstm_1[0][0]                     
                                                                   lstm_1[0][2]                     
                                                                   dot_1[2][0]                      
                                                                   lstm_1[1][0]                     
                                                                   lstm_1[1][2]                     
                                                                   dot_1[3][0]                      
                                                                   lstm_1[2][0]                     
                                                                   lstm_1[2][2]                     
                                                                   dot_1[4][0]                      
                                                                   lstm_1[3][0]                     
                                                                   lstm_1[3][2]                     
                                                                   dot_1[5][0]                      
                                                                   lstm_1[4][0]                     
                                                                   lstm_1[4][2]                     
                                                                   dot_1[6][0]                      
                                                                   lstm_1[5][0]                     
                                                                   lstm_1[5][2]                     
                                                                   dot_1[7][0]                      
                                                                   lstm_1[6][0]                     
                                                                   lstm_1[6][2]                     
                                                                   dot_1[8][0]                      
                                                                   lstm_1[7][0]                     
                                                                   lstm_1[7][2]                     
                                                                   dot_1[9][0]                      
                                                                   lstm_1[8][0]                     
                                                                   lstm_1[8][2]                     
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 11)            1419        lstm_1[0][0]                     
                                                                   lstm_1[1][0]                     
                                                                   lstm_1[2][0]                     
                                                                   lstm_1[3][0]                     
                                                                   lstm_1[4][0]                     
                                                                   lstm_1[5][0]                     
                                                                   lstm_1[6][0]                     
                                                                   lstm_1[7][0]                     
                                                                   lstm_1[8][0]                     
                                                                   lstm_1[9][0]                     
====================================================================================================
Total params: 185,484
Trainable params: 185,484
Non-trainable params: 0
____________________________________________________________________________________________________

'''


### START CODE HERE ### (≈2 lines)### STAR
opt = Adam(lr = 0.005, beta_1=0.9, beta_2=0.999, decay = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
### END CODE HERE ###


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)