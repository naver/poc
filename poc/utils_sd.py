def ensure_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

def even_list_len(x, n):
    if len(x) == 1:
        x = x * n
    elif len(x) != n:
        raise ValueError(f'List longer than 1 with length different than {n}')
    return x
    