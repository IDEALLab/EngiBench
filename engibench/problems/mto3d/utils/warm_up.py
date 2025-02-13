import os, shutil
import zipfile
import argparse

template_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'templates'
)

def change_continuation(contents, continuation):
    for i, line in enumerate(contents):
        if line == 'continuation\n' and contents[i+1] == '{\n':
            cont_arg = '    NoContinuation      {};\n'.format(int(not continuation))
            if 'NoContinuation' in contents[i+2]:
                contents[i+2] = cont_arg
            else:
                contents.insert(i+2, cont_arg)
            return

def change_continuation_parameter(contents, parameter, _from=None, _to=None):
    if not (_from is None and _to is None):
        for i, line in enumerate(contents):
            if line == '    {}\n'.format(parameter) and contents[i+1] == '    {\n':
                if _from is not None:
                    contents[i+4] = '        from    {};\n'.format(_from)
                if _to is not None:
                    contents[i+5] = '        to    {};\n'.format(_to)
                return

def change_continuation_setting(contents, setting, value):
    if value is not None:
        for i, line in enumerate(contents):
            if setting in line:
                contents[i] = '    {}      {};\n'.format(setting, value)
                return

def change_condition(contents, condition, value):
    if value is not None:
        for i, line in enumerate(contents):
            if condition in line:
                contents[i] = '{}{};\n'.format(condition.ljust(20), value)
                return
            
def change_U(contents, value):
    if value is not None:
        for i, line in enumerate(contents):
            if 'inlet' in line:
                temp = '        value           uniform (0 0 -{});\n'
                contents[i+3] = temp.format(value)
                return

def modify_optProperties(
        path, 
        continuation=True, 
        qU: tuple=(None, None),
        alphaMax: tuple=(None, None),
        Heaviside: tuple=(None, None),
        PowerDissMax: float=None, 
        voluse: float=None,
        U: float=None,
        n_beginning: int=None,
        n_levels: int=None,
        n_transitionItersAtEachLevel: int=None,
        n_restPeriodAtEachLevel: int=None,
        n_maxIterAtEach: int=None,
        n_consecutiveConverged: int=None,
        **kwargs
        ):
    optPrpty = os.path.join(template_path, 'optProperties_')
    with open(optPrpty, "r") as f: contents = f.readlines()[40:] # only after 40 lines
    change_continuation(contents, continuation)
    change_continuation_parameter(contents, 'qU', *qU)
    change_continuation_parameter(contents, 'alphaMax', *alphaMax)
    change_continuation_parameter(contents, 'Heaviside', *Heaviside)
    
    change_continuation_setting(contents, 'n_beginning', n_beginning)
    change_continuation_setting(contents, 'n_levels', n_levels)
    change_continuation_setting(contents, 'n_transitionItersAtEachLevel', n_transitionItersAtEachLevel)
    change_continuation_setting(contents, 'n_restPeriodAtEachLevel', n_restPeriodAtEachLevel)
    change_continuation_setting(contents, 'n_maxIterAtEach', n_maxIterAtEach)
    change_continuation_setting(contents, 'n_consecutiveConverged', n_consecutiveConverged)

    # change PowerDissMax and voluse
    opt_path = os.path.join(path, 'constant', 'optProperties')
    with open(opt_path, "r") as f: conditions = f.readlines()[:40]
    change_condition(conditions, 'PowerDissMax', PowerDissMax)
    change_condition(conditions, 'voluse', voluse)
    
    text = ''.join(conditions + contents) 
    with open(opt_path, "w") as f: f.write(text)

    # change U
    U_path = os.path.join(path, '0', 'U')
    with open(U_path, "r") as f: field = f.readlines()
    change_U(field, U)
    with open(U_path, "w") as f: f.write(field)

def replace_src(path):
    zip = os.path.join(template_path, "warm-ready-3d.zip")
    with zipfile.ZipFile(zip, mode="r") as archive:
        archive.extractall(os.path.join(path, 'src'))

def add_xh(path, xh):
    path_0 = os.path.join(path, '0', 'xh.gz')
    shutil.copyfile(xh, path_0)

def initialize(
        path, xh, 
        continuation, 
        qU, alphaMax, Heaviside, 
        PowerDissMax, voluse, U,
        n_beginning, n_levels, n_transitionItersAtEachLevel,
        n_restPeriodAtEachLevel, n_maxIterAtEach, n_consecutiveConverged
        ):
    assert os.path.exists(path)
    modify_optProperties(
        path, 
        continuation, 
        qU, alphaMax, Heaviside,
        PowerDissMax, voluse, U,
        n_beginning, n_levels, n_transitionItersAtEachLevel, 
        n_restPeriodAtEachLevel, n_maxIterAtEach, n_consecutiveConverged
        )
    replace_src(path)
    add_xh(path, xh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='target path')
    parser.add_argument('xh', help='path of xh')
    
    parser.add_argument('-c', '--continuation', type=int, default=0, help='continuation switch')
    parser.add_argument('--qU', type=eval, default='(0.01,0.001)', help='controls interpolation')
    parser.add_argument('--alphaMax', type=eval, default='(2500.0,2.5e6)', help='controls Darcy number')
    parser.add_argument('--Heaviside', type=eval, default='(1,100.0)', help='controls Heaviside')

    parser.add_argument('--PowerDissMax', type=eval, default='None', help='upper bound of power dissipation')
    parser.add_argument('--voluse', type=eval, default='None', help='volume fraction of fluid channels')
    parser.add_argument('--U', type=eval, default='None', help='inlet fluid velocity')

    parser.add_argument('--n_beginning', type=int, default=20, help='the number of iter at the first level')
    parser.add_argument('--n_levels', type=int, default=20, help='the total numebr of levels') # this needs to be modifed if a different continuation is used.
    parser.add_argument('--n_transitionItersAtEachLevel', type=int, default=20, help='the number of transition iter between each level')
    parser.add_argument('--n_restPeriodAtEachLevel', type=int, default=10, help='the least number of iter at each level')
    parser.add_argument('--n_maxIterAtEach', type=int, default=100, help='the maximum number of iter at each level')
    parser.add_argument('--n_consecutiveConverged', type=int, default=2, help='the number of iter at the end')

    args = parser.parse_args()

    initialize(
        args.path, args.xh, 
        args.continuation, 
        args.qU, args.alphaMax, args.Heaviside,
        args.PowerDissMax, args.voluse, args.U,
        args.n_beginning, args.n_levels, args.n_transitionItersAtEachLevel,
        args.n_restPeriodAtEachLevel, args.n_maxIterAtEach, args.n_consecutiveConverged
        )

    # test code: python warm_start/3D/warm_up.py instances/3D/3Dheatsink_0/ instances/test/xh0.gz 