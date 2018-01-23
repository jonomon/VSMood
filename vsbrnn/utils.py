import numpy as np

def makeGaussian(size, centre, fwhm=1):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)[:,np.newaxis]
    x0 = centre[0]
    y0 = centre[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def isPointInRect(self, px, py):
        if px > self.x and px < (self.x + self.w):
            within_x = True
        else:
            within_x = False
        if py > self.y and py < (self.y + self.h):
            within_y = True
        else:
            within_y = False
        return within_x and within_y

def get_max_sequence_len(sequence):
    max_len = 0
    for i in sequence:
        if len(i) > max_len:
            max_len = len(i)
    return max_len

def tokenise_sequence(sequence, model):
    token_4 = {"LE": 1, "RE": 2, "NO": 3, "M": 4, "NT": 5, "LC": 6, "RC": 7, "FH1": 8,
               "FH2": 9, "FH3": 10, "LM": 11, "RM": 12}
    token_grid9 = {"FH1": 1, "FH2": 2, "FH3": 3, "LE": 4, "NT": 5, "RE": 6, "LC": 7, "NO": 8,
                   "RC": 9}
    token_grid16 = {"FH1": 1, "FH2": 2, "FH3": 3, "FH4": 4, "LE1": 5, "LE2": 6, "RE1": 7,
                    "RE2": 8, "LC": 9, "NO1": 10, "NO2": 11, "RC": 12, "M1": 13, "M2": 14,
                    "M3": 15, "M4": 16}

    token_semantic5 = {"LE": 1, "RE": 2, "NO": 3, "M": 4}
    token_semantic8 = {"FH": 1, "LE": 2, "RE": 3, "NO": 4, "LC": 5, "RC": 6, "M": 7}

    if sorted(token_4.keys()) == sorted(model.model_TGH.keys()):
        token = token_4
    if sorted(token_grid9.keys()) == sorted(model.model_TGH.keys()):
        token = token_grid9
    if sorted(token_semantic5.keys()) == sorted(model.model_TGH.keys()):
        token = token_semantic5
    if sorted(token_grid16.keys()) == sorted(model.model_TGH.keys()):
        token = token_grid16
    if sorted(token_semantic8.keys()) == sorted(model.model_TGH.keys()):
        token = token_semantic8

    assert token is not None, "Token incorrect"
    token["N"] = len(token) + 1
    out = []
    for i in sequence:
        seq = []
        for v in i:
            if isinstance(v, int):
                seq.append(v)
            else:
                seq.append(token[v])
        out.append(seq)
    return out

def tokenise_cat(cat, one_hot=False):
    token = {"BD": 0, "BR": 0, "C": 1, "D": 1, "R": 1}
    out = []
    for i in cat:
        if one_hot:
            if token[i] == 1:
                out_i = [0, 1]
            else:
                out_i = [1, 0]
            out.append(out_i)
        else:
            out.append(token[i])
    return np.array(out)


def tokenise_img_type(img_types):
    token = {"Happy": 2, "Sad": 1}
    out = []
    for img_label in img_types:
        out.append([token[a] for a in img_label])
    return out

def get_sub_to_cat_dict(sub, cat):
    sub_to_cat = {}
    for s, c in zip(sub, cat):
        if s not in sub_to_cat:
            sub_to_cat[s] = c
    return sub_to_cat

def _getProb(i, n_, bins):
    n_ = n_ / np.sum(n_)
    cum = filter(lambda a: i >= a[1], zip(n_, bins))
    if len(cum) == 0:
        n_curr = n_[0]
    else:
        n_curr = cum[-1][0]
    n_curr = np.clip(n_curr, 0.001, 1)
    return n_curr
        
def get_log_likelihood(i, n_bd, bins_bd, n_d, bins_d):
    p_bd = _getProb(i, n_bd, bins_bd)
    p_d = _getProb(i, n_d, bins_d)
    log_p_d = np.log(p_d)
    log_p_bd = np.log(p_bd)
    log_like_bd_d = log_p_d - log_p_bd
    return log_like_bd_d
