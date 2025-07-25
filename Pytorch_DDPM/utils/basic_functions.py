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

def num_to_groups(num, divisior):
    groups = num // divisior
    remainder = num % divisior
    arr = [divisior] * groups
    if remainder>0:
        arr.append(remainder)
    return arr

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
