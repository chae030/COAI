def sigmoid (x) :
    return 1 / (1 + np.exp(-x))

def loss_func (x, t) :
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))

def numerical_derivative (f, x) :
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished :
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        
        x[idx] = tmp_val
        it.iternext()
        
    return grad

def error_val (x, t) :
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return np.sum (t * np.log(y + delta) + (1 + t) * np.log((1 - y) + delta))

def predict(x) :
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    if y >= 0.5 :
        result = 1
    else :
        result = 0
    return y, result