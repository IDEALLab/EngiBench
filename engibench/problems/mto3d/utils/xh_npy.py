import os
import gzip
import numpy as np

de1 = '264000\n(\n'
de2 = '\n)\n'

def gamma_to_tensor(gamma):
    # main body #1
    gamma_MB1_b1 = np.reshape(gamma[20000:21600], (40,20,2))
    gamma_MB1_b2 = np.reshape(gamma[21600:84000], (40,20,78))
    gamma_MB1_b3 = np.reshape(gamma[84000:100000], (40,20,20))
    gamma_MB1_b = np.concatenate((gamma_MB1_b1, gamma_MB1_b2, gamma_MB1_b3), axis=2)
    # main body #2, 0 in between
    gamma_MB2_b1 = np.reshape(gamma[112000:112800], (20,20,2))
    gamma_MB2_b2 = np.reshape(gamma[112800:144000], (20,20,78))
    gamma_MB2_b3 = np.reshape(gamma[144000:152000], (20,20,20))
    gamma_MB2_b = np.concatenate((gamma_MB2_b1, gamma_MB2_b2, gamma_MB2_b3), axis=2)
    # main body #3, 0 in between
    gamma_MB3_b1 = np.reshape(gamma[164000:165600], (40,20,2))
    gamma_MB3_b2 = np.reshape(gamma[165600:228000], (40,20,78))
    gamma_MB3_b3 = np.reshape(gamma[228000:244000], (40,20,20))
    gamma_MB3_b = np.concatenate((gamma_MB3_b1, gamma_MB3_b2, gamma_MB3_b3), axis=2)
    # concatenate to construct half of the design domain
    gamma_MB = np.concatenate((gamma_MB1_b, gamma_MB2_b, gamma_MB3_b), axis=0)
    # mirror and concatenate to make the entire design
    gamma_MB_full = np.flipud(np.concatenate((gamma_MB, np.flip(gamma_MB, 2)), axis=2))
    return gamma_MB_full

def tensor_to_gamma(tensor):
    gamma_MB_full = np.flipud(tensor)
    gamma_MB = np.split(gamma_MB_full, 2, axis=2)[0]
    gamma_MB1_b, gamma_MB2_b, gamma_MB3_b = np.split(gamma_MB, [40, 60])

    gamma_MB1 = np.split(gamma_MB1_b, [2, 80], axis=2)
    gamma_MB2 = np.split(gamma_MB2_b, [2, 80], axis=2)
    gamma_MB3 = np.split(gamma_MB3_b, [2, 80], axis=2)
    
    gamma = [b.flatten() for b in gamma_MB1] + [np.zeros(12000)] \
          + [b.flatten() for b in gamma_MB2] + [np.zeros(12000)] \
          + [b.flatten() for b in gamma_MB3]
    return np.concatenate(gamma)

def read_xh(path):
    with gzip.open(path, 'rt', newline='\n') as f:
        file_content = f.read()
    head, body = file_content.split(de1)
    body, tail = body.split(de2)
    return head, body, tail
    
def xh_to_npy(path):
    _, field, _ = read_xh(path)
    gamma = np.asarray(field.split('\n'), dtype=float)
    tensor = gamma_to_tensor(gamma)
    return tensor.transpose((1, 0, 2))

def npy_to_xh(tensor, path, name='xh.gz', template=None): # tensor shape (c, h, w) = (20, 100, 200)
    if template is None:
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'templates'
        )
        template = os.path.join(template_path, 'xh_template.gz')
    head, field, tail = read_xh(template) # location    "1922";
    head = head.replace('location    "1922";', 'location    "0";')
    gamma = np.asarray(field.split('\n'), dtype=float)
    tensor = tensor.transpose((1, 0, 2))
    gamma[20000:-20000] = tensor_to_gamma(tensor)

    np.savetxt(
        os.path.join(path, name), 
        gamma, '%.2e', 
        header=''.join([head, de1[:-1]]),
        footer=''.join([de2[1:], tail]),
        comments=''
        )