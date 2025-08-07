from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    if isfunction(d):
        return d()
    else:
        return d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)

    target_shape = [batch_size] + [1] * (len(x_shape) - 1)
    out = out.reshape(target_shape).to(t.device)
    return out
