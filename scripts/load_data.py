from os.path import join, exists
from time import time
import numpy as np

PATH = 'data'

def to_float(x):
    x = np.asarray(x)
    try:
        z = x.astype(float).reshape((len(x), 1))
    except:
        names = np.unique(x)
        z = np.zeros((len(x), len(names) - 1))
        for j in range(1, len(names)):
            z[x == names[j], j - 1] = 1
    return z


def load_uci(name, delimiter=','):
    fname = join(PATH, name + '.data')
    if not exists(fname):
        fname = join(PATH, name + '.csv')
    return np.loadtxt(fname, dtype=str, delimiter=delimiter)


def get_data_ionosphere():
    dat = load_uci('ionosphere')
    aux = dat[:, -1]
    y = -np.ones(len(aux))
    y[aux == 'g'] = 1
    x = dat[:, 0:-1].astype(float)
    xdesc = []
    for i in range(x.shape[1] / 2):
        xdesc += [str(i + 1) + 'r', str(i + 1) + 'c']
    xdesc = np.array(xdesc)
    """
    Remove first pulse number, which looks weird...
    """
    msk = range(x.shape[1])
    msk.remove(0)
    msk.remove(1)
    x = x[:, msk]
    xdesc = xdesc[msk]
    return y, x, xdesc


def get_data_parkinsons():
    dat = load_uci('parkinsons')
    cap = dat[0, :]
    dat = dat[1:, :]
    idx = np.where(cap == 'status')[0]
    y = 2 * dat[:, idx].astype(int).squeeze() - 1
    msk = range(len(cap))
    msk.remove(idx)
    msk.remove(np.where(cap == 'name')[0])
    x = dat[:, msk].astype(float)
    xdesc = cap[msk]
    return y, x, xdesc


def get_data_haberman():
    dat = load_uci('haberman')
    xdesc = np.asarray(('age', 'year_of_operation', 'auxillary_nodes'))
    x = dat[:, 0:3].astype(float)
    y = 2 * dat[:, -1].astype(int) - 3
    return y, x, xdesc


def get_data_SPECTF():
    dat = load_uci('SPECTF.train')
    y = 2 * dat[:, 0].astype(int) - 1
    x = dat[:, 1:].astype(float)
    xdesc = np.array(['R' + str(i + 1) for i in range(x.shape[1])])
    return y, x, xdesc


def get_data_wpbc():
    dat = load_uci('wpbc')
    x = dat[:, 2:]
    msk = np.min(x != '?', 1)
    x = x[msk, :].astype(float)
    y = 2 * (dat[msk, 1] == 'R').astype(int) - 1
    xdesc = np.array(['f' + str(i + 1) for i in range(x.shape[1])])
    return y, x, xdesc


def get_data_wdbc():
    dat = load_uci('wdbc')
    x = dat[:, 2:].astype(float)
    y = 2 * (dat[:, 1] == 'M').astype(int) - 1
    xdesc = np.array(['f' + str(i + 1) for i in range(x.shape[1])])
    return y, x, xdesc


def get_data_bank():
    dat = load_uci('bank-full', delimiter=';')
    y = 2 * (dat[1:, -1] == '"yes"') - 1
    xdesc = np.array([d.replace('"', '') for d in dat[0, 0:16]])
    dat = dat[1:, 0:16]
    x = []
    for j in range(dat.shape[1]):
        x.append(to_float(dat[:, j]))
    x = np.concatenate(x, axis=1)
    return y, x, xdesc


def get_data_adult():
    dat = load_uci('adult')
    y = 2 * (dat[:, -1] == ' >50K') - 1
    dat = dat[:, 0:-1]
    x = []
    for j in range(dat.shape[1]):
        x.append(to_float(dat[:, j]))
    x = np.concatenate(x, axis=1)
    return y, x, None


def get_data(key):
    if key == 'ionosphere':
        return get_data_ionosphere()
    elif key == 'parkinsons':
        return get_data_parkinsons()
    elif key == 'haberman':
        return get_data_haberman()
    elif key == 'SPECTF':
        return get_data_SPECTF()
    elif key == 'wpbc':
        return get_data_wpbc()
    elif key == 'wdbc':
        return get_data_wdbc()
    elif key == 'bank':
        return get_data_bank()
    elif key == 'adult':
        return get_data_adult()
    else:
        raise ValueError('unknown dataset')

    
