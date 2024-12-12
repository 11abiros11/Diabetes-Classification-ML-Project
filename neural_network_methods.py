import numpy as np
import tensorflow as tf

def initialize_params(layer_array, X, weight_factor):
    
#     np.random.seed(3)
    params = {}
    
    for l in range(len(layer_array)):
        if(l==0):
            W = np.random.randn(layer_array[l], X.shape[0])*weight_factor
        else:
            W = np.random.randn(layer_array[l], layer_array[l-1])*weight_factor
        
        b = np.zeros((layer_array[l],1))

        params[(f"W{l+1}")] = W
        params[(f"b{l+1}")] = b
    return params

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_prop(params, activation_array, layer_array, X):
    
    A = X
    stored_Al_Zl = {}
#     print(len(layer_array))
    for l in range(len(layer_array)):
        
        W = params[f"W{l+1}"]
        b = params[f"b{l+1}"]
        
        Z = np.dot(W,A) + b
#         print(l, Z.shape, W.shape, A.shape, (np.dot(W,A)).shape, b.shape)
        if(activation_array[l] == 'Relu'):
            A = tf.keras.activations.relu(Z).numpy()
        else:
#             Z_normalize = (Z- np.mean(Z))/np.std(Z)
            A = sigmoid(Z)
        
        stored_Al_Zl[(f"A{l+1}")] = A
        stored_Al_Zl[(f"Z{l+1}")] = Z
        
    return A, stored_Al_Zl   

def comp_cost(Y, A_final):
    
    m = A_final.shape[1]

#     print(Y * np.log(A_final))
#     print((1 - Y) * np.log(1 - A_final))
    cost = Y * np.log(A_final) + (1 - Y) * np.log(1 - A_final)
#     cost = np.squeeze(cost)
    return -np.sum(cost)/m

def relu_backprop(Z, operation, cond, cond2):
    
    dZ = Z

    dZ[cond]   = 1
    dZ[cond2]  = operation(Z[cond2])

    return dZ

def back_prop(layer_array, activation_array, stored_Al_Zl, params, X, Y):
    
    grads = {}
    #####################################
    #OUTPUT LAYER-> SIGMOID ACTIVATION FN
    #####################################
    
    L = len(layer_array)
    m = X.shape[1]
    
    A_L = stored_Al_Zl[f"A{L}"]
    
    if(L>1):
        A_L_prev = stored_Al_Zl[f"A{L-1}"]
    else:
        A_L_prev = X
        
    Z_L = stored_Al_Zl[f"Z{L}"]
        
    dZ = A_L - Y
    dW = (1/m) * np.dot(dZ, A_L_prev.T) 
    db = (1/m) * np.sum(dZ, axis = 1, keepdims= True)
    
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db
    
    for l in reversed(range(len(layer_array)-1)):
#         print('backprop:',l)
        A_l = stored_Al_Zl[f"A{l+1}"]
        Z_l = stored_Al_Zl[f"Z{l+1}"]
        
        W_l = params[f"W{l+1}"]
#         print(l+1)
        
        ######################################
        #HIDDEN LAYERS
        ######################################
        
        if(l>0):
            A_prev = stored_Al_Zl[f"A{l}"]
        else:
            A_prev = X
            
        W_l_plus1 = params[f"W{l+2}"]   
        dZ_l_plus1 = dZ
        dA_dZ = relu_backprop(Z_l, lambda x: x==0,  Z_l>0, Z_l<=0)
        
        dZ = dA_dZ * np.dot(W_l_plus1.T, dZ_l_plus1)

        dW = (1/m) * np.dot(dZ, A_prev.T) 
        db = (1/m) * np.sum(dZ, axis = 1, keepdims= True)
#         dA_prev = np.dot(W_l.T, dZ)

        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db
            
#         if(activation_array[l] == 'Relu'):
# #             print(l,'relu')
#             W_prev = params[f"W{l+2}"]   
#             dZ_prior = relu_backprop(Z_l, lambda x: x==0,  Z_l>0, Z_l<=0)
# #             print(Z_l, dZ_prior)
#             dZ = dZ_prior * np.dot(W_prev.T,)
            
#             dW = (1/m) * np.dot(dZ, A_prev.T) 
#             db = (1/m) * np.sum(dZ, axis = 1, keepdims= True)
#             dA_prev = np.dot(W_l.T, dZ)
            
#             grads[f"dW{l+1}"] = dW
#             grads[f"db{l+1}"] = dW
#         else:
# #             print(l,'sigmoid')
#             dZ = A_l - Y
            
#             dW = (1/m) * np.dot(dZ, A_prev.T) 
#             db = (1/m) * np.sum(dZ, axis = 1, keepdims= True)
#             dA_prev = np.dot(W_l.T, dZ)
            
#             grads[f"dW{l+1}"] = dW
#             grads[f"db{l+1}"] = dW
            
    return grads

def update_params(params, grads, alpha, layer_array):
    
    for l in range(len(layer_array)):
        params[(f"W{l+1}")] = params[(f"W{l+1}")] - alpha * grads[f"dW{l+1}"]
        params[(f"b{l+1}")] = params[(f"b{l+1}")] - alpha * grads[f"db{l+1}"]
    
    return params

def predict(params, activation_array, layer_array, X, Y):
    
    A, stored_Al_Zl = forward_prop(params, activation_array, layer_array, X)
    
    pred = A>.5
    
    count = 0
    for i in range(len(Y)):
        if(pred[0][i]-Y[i]==0):
            count +=1
    
    accuracy = count/len(Y)
    return pred, A, accuracy