import numpy as np
import tensorflow as tf
import copy
def initialize_params(layer_array, X, weight_factor, method = 'He'):
    
    np.random.seed(5)
    params = {}
    
    for l in range(len(layer_array)):
        if(method == 'He'):
            if(l==0):
                W = np.random.randn(layer_array[l], X.shape[0])*weight_factor * np.sqrt(2/(X.shape[0]))
            else:
                W = np.random.randn(layer_array[l], layer_array[l-1])*weight_factor * np.sqrt(2/(layer_array[l-1]))
        else:
            if(l==0):
                W = np.random.randn(layer_array[l], X.shape[0])*weight_factor
            else:
                W = np.random.randn(layer_array[l], layer_array[l-1])*weight_factor
                
        b = np.zeros((layer_array[l],1))

        params[(f"W{l+1}")] = W
        params[(f"b{l+1}")] = b
    return params

def initialize_optimizer(params, layer_array, method = 'momentum'):
    
    L = len(layer_array)
    v = {}
    s = {}
    
    if(method == 'momentum'):
        for l in range(L):
            v[(f'dW{l+1}')] = np.zeros_like(params[f'W{l+1}'])
            v[(f'db{l+1}')] = np.zeros_like(params[f'b{l+1}'])

        return v

    elif(method == 'Adam'):
        for l in range(L):
            v[(f'dW{l+1}')] = np.zeros_like(params[f'W{l+1}'])
            v[(f'db{l+1}')] = np.zeros_like(params[f'b{l+1}'])

            s[(f'dW{l+1}')] = np.zeros_like(params[f'W{l+1}'])
            s[(f'db{l+1}')] = np.zeros_like(params[f'b{l+1}'])
            
        return v, s

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

def comp_cost(Y, A_final, params, L, lamb, regularization = True):
    
    m = A_final.shape[1]

    cost = Y * np.log(A_final) + (1 - Y) * np.log(1 - A_final)
    
    reg_cost = 0
    if(regularization):
        for l in range(L):
            W_l = params[f"W{l+1}"]
            reg_cost += (1/m) * (lamb/2) * np.sum(np.square(W_l))

    return -np.sum(cost)/m + reg_cost

def relu_backprop(Z, operation, cond, cond2):
    
    dZ = Z.copy()

    dZ[cond]   = 1
    dZ[cond2]  = operation(Z[cond2])

    return dZ

# def gradient_check(activation_fn):
    
#     if(activation_fn == 'sigmoid'):
        
    

def back_prop(layer_array, activation_array, stored_Al_Zl, params, X, Y, lamb, regularization = True):
    
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
    
    W_L = params[f"W{L}"]
    if(regularization):
        dW = (1/m) * np.dot(dZ, A_L_prev.T) + (lamb/m) * W_L
    else:
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
        
        dL_dA = np.matmul(W_l_plus1.T, dZ_l_plus1)
        dA_dZ = relu_backprop(Z_l, lambda x: x==0,  Z_l>0, Z_l<=0)
        dZ    = dL_dA * dA_dZ
        
        if(regularization):
            dW = (1/m) * np.dot(dZ, A_prev.T) + (lamb/m)*W_l 
        else:
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

def update_params(params, grads, alpha, layer_array, v, s, t, beta_momentum, beta_adam_1, beta_adam_2, eps = 1e-8, method = 'normal'):
    
    v_fixed = {}
    s_fixed = {}

    if(method == 'normal'):
        for l in range(len(layer_array)):
            params[(f"W{l+1}")] = params[(f"W{l+1}")] - alpha * grads[f"dW{l+1}"]
            params[(f"b{l+1}")] = params[(f"b{l+1}")] - alpha * grads[f"db{l+1}"]
            
#             print(grads[f"dW{l+1}"], grads[f"db{l+1}"])
        return params
    
    elif(method == 'momentum'):
        for l in range(len(layer_array)):
            v[(f"dW{l+1}")] = beta_momentum * v[(f"dW{l+1}")] + (1-beta_momentum) * grads[f"dW{l+1}"]
            v[(f"db{l+1}")] = beta_momentum * v[(f"db{l+1}")] + (1-beta_momentum) * grads[f"db{l+1}"]

            params[(f"W{l+1}")] = params[(f"W{l+1}")] - alpha * v[(f"dW{l+1}")]
            params[(f"b{l+1}")] = params[(f"b{l+1}")] - alpha * v[(f"db{l+1}")]
            
#             print(grads[f"dW{l+1}"], grads[f"db{l+1}"])
        return params, v
    
    elif(method == 'Adam'):
        for l in range(len(layer_array)):
            v[(f"dW{l+1}")] = beta_adam_1 * v[(f"dW{l+1}")] + (1-beta_adam_1) * grads[f"dW{l+1}"]
            v[(f"db{l+1}")] = beta_adam_1 * v[(f"db{l+1}")] + (1-beta_adam_1) * grads[f"db{l+1}"]

            v_fixed[(f"dW{l+1}")] = v[(f"dW{l+1}")]/(1-beta_adam_1**t)
            v_fixed[(f"db{l+1}")] = v[(f"db{l+1}")]/(1-beta_adam_1**t)

            s[(f"dW{l+1}")] = beta_adam_1 * s[(f"dW{l+1}")] + (1-beta_adam_1) * grads[f"dW{l+1}"]
            s[(f"db{l+1}")] = beta_adam_1 * s[(f"db{l+1}")] + (1-beta_adam_1) * grads[f"db{l+1}"]

            s_fixed[(f"dW{l+1}")] = s[(f"dW{l+1}")]/(1-beta_adam_2**t)
            s_fixed[(f"db{l+1}")] = s[(f"db{l+1}")]/(1-beta_adam_2**t)

            params[(f"W{l+1}")] = params[(f"W{l+1}")] - alpha * v_fixed[(f"dW{l+1}")]/(np.sqrt(s_fixed[(f"dW{l+1}")]) + eps)
            params[(f"b{l+1}")] = params[(f"b{l+1}")] - alpha * v_fixed[(f"db{l+1}")]/(np.sqrt(s_fixed[(f"db{l+1}")]) + eps)
            
#             print(beta_adam_1, grads[f"dW{l+1}"], grads[f"db{l+1}"])
        return params, v, s, v_fixed, s_fixed

def predict(params, activation_array, layer_array, X, Y):
    
    A, stored_Al_Zl = forward_prop(params, activation_array, layer_array, X)
    
    pred = A>.5
    
    count = 0
    for i in range(len(Y)):
        if(pred[0][i]-Y[i]==0):
            count +=1
    
    accuracy = count/len(Y)
    return pred, A, accuracy