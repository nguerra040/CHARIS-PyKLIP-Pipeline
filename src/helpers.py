import os

def boolean(string):
    if string == 'True' or string == 'true':
        return True
    elif string == 'False' or string == 'false':
        return False
    elif string == 'None' or string == 'none':
        return None
    else:
        raise Exception('expect true or false string.')

def get_bash_path(p):
    path = p.replace(' ', '\\ ')
    return path
