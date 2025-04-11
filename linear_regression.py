def loss_func (x, t) :
    y = np.dot(x, W) + b
    return (np.sum((t - y) ** 2)) / (len(x))

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
    y = np.dot (x, W) + b
    return (np.sum((t - y) ** 2)) / (len(x))

def predict (x) :
    y = np.dot(x, W) + b
    return y