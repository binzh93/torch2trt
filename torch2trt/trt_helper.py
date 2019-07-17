

def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

    
def check_type(vars, types):
    for var, t in zip(vars, types):
        if not isinstance(var, t):
            return False
    return True


def check_shape(vars, expected):
    for var, e in zip(vars, expected):
        if var.shape != e:
            return False
    return True


