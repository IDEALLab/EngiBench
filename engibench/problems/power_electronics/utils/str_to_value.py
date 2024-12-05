import re


def str_to_value(s):
    # turns '10kF'(str) to 1e4(float).
    try:
        value = float(s)
    except:
        mappings = {"u": 1e-6, "U": 1e-6, "m": 1e-3, "M": 1e-3, "k": 1e3, "K": 1e3, "meg": 1e6}
        value, s = (i for i in re.split(r"([A-Za-z]+)", s) if i)  # https://stackoverflow.com/a/28290501
        value = float(value) * mappings[s]

    return value
