import numpy as np
import argparse
import LammpsH5MD
import msd
import isf
import contactmap
import distmap
import rdp
import sdp
import sys
import yaml
import datetime

# ===================================================================
# define a dictionary for function name look up
func_name_dic = {
    'msd.g1': msd.g1,
    'msd.g2': msd.g2,
    'msd.g3': msd.g3,
    'isf':    isf.isf,
    'cmap': contactmap.contactmap,
    'rdp': rdp.rdp,
    'sdp': sdp.sdp,
    'dmap': distmap.distmap
}

twotime_func_name_lst = ['msd.g1', 'msd.g2', 'msd.g3', 'isf']
onetime_func_name_lst = ['cmap', 'rdp', 'sdp', 'dmap']
# ===================================================================

# ===================================================================
# function read_parameter takes the yaml script as input,
# return
#   finname: path to the yaml format parameters file
#   func_lst: the function list which feed into LammpsH5MD.cal_correlate()

twotime_flag = 0
onetime_flag = 0

def read_parameter(script):
    twofunc_dic = {}
    onefunc_dic = {}
    write_dic = {}

    global twotime_flag
    global onetime_flag

    for key, value in script.iteritems():
        if key == 'FILE':
            finname = value
        elif key == 'COMPUTE':
            func_name_lookup = {}
            for func in value:
                for func_name, info in func.iteritems():
                    func_name_lookup[info['id']] = func_name
                    if func_name not in twotime_func_name_lst and func_name not in onetime_func_name_lst:
                        raise NameError('Unknown function {}. Supported function: {}\n'.format(func_name, func_name_dic.keys()))
                    if 'args' in info:
                        args = info['args']
                        if func_name in twotime_func_name_lst:
                            twofunc_dic[info['id']] = func_name_dic[func_name](**args)
                        elif func_name in onetime_func_name_lst:
                            onefunc_dic[info['id']] = func_name_dic[func_name](**args)
                    else:
                        args = None
                        if func_name in twotime_func_name_lst:
                            twofunc_dic[info['id']] = func_name_dic[func_name]
                        elif func_name in onetime_func_name_lst:
                            onefunc_dic[info['id']] = func_name_dic[func_name]
        elif key == 'WRITE':
            for write in value:
                for write_name, info in write.iteritems():
                    write_dic[info['id']] = write_name

    if 'ARGS_TWOTIME' in script and 'ARGS_ONETIME' not in script:
        twotime_flag = 1
        return [finname, script['ARGS_TWOTIME'], func_name_lookup, write_dic, twofunc_dic]
    elif 'ARGS_ONETIME' in script and 'ARGS_TWOTIME' not in script:
        onetime_flag = 1
        return [finname, script['ARGS_ONETIME'], func_name_lookup, write_dic, onefunc_dic]
    elif 'ARGS_ONETIME' in script and 'ARGS_TWOTIME' in script:
        twotime_flag, onetime_flag = 1, 1
        return [finname, script['ARGS_TWOTIME'], script['ARGS_ONETIME'], func_name_lookup, write_dic, twofunc_dic, onefunc_dic]
    else:
        raise ValueError('Please specify the argument passed to COMPUTE\n')
# ===================================================================


# parse the arguments
# usage:
#   python AnalysisTool.py parameter_script.txt -q
#   first argument: script parameter file.
#   --quite(-q): not output the information on screeen

parser = argparse.ArgumentParser(description='H5MD trajectory file analysis tool')
parser.add_argument('parameter_file', help='parameter script file')
parser.add_argument('-q', '--quite', help='enable/disable quite execution', action='store_false', default=True, dest='screen_info')
args = parser.parse_args()

# parameters file is written use YAML syntax
# load the parameters file using yaml
with open(args.parameter_file, 'r') as f:
    parameters = yaml.load(f)

# Get the parameters
parameters_result_lst = read_parameter(parameters)
if twotime_flag == 1 and onetime_flag == 0:
    finname = parameters_result_lst[0]
    twotime_kwargs = parameters_result_lst[1]
    func_name_lookup = parameters_result_lst[2]
    write_dic = parameters_result_lst[3]
    twofunc_dic = parameters_result_lst[4]
elif twotime_flag == 0 and onetime_flag == 1:
    finname = parameters_result_lst[0]
    onetime_kwargs = parameters_result_lst[1]
    func_name_lookup = parameters_result_lst[2]
    write_dic = parameters_result_lst[3]
    onefunc_dic = parameters_result_lst[4]
elif twotime_flag == 1 and onetime_flag == 1:
    finname = parameters_result_lst[0]
    twotime_kwargs = parameters_result_lst[1]
    onetime_kwargs = parameters_result_lst[2]
    func_name_lookup = parameters_result_lst[3]
    write_dic = parameters_result_lst[4]
    twofunc_dic = parameters_result_lst[5]
    onefunc_dic = parameters_result_lst[6]


traj = LammpsH5MD.LammpsH5MD()    # create LammpsH5MD class
traj.load(finname)     # load the trajectory

# do the calculation
# two-time quantity calculation
if twotime_flag == 1:
    twotime_data_dic = traj.cal_twotime(twofunc_dic.values(), screen_info=args.screen_info, **twotime_kwargs)
    twotime_output_name_lst = []
    for key in twofunc_dic.keys():
        twotime_output_name_lst.append(write_dic[key])
    for key in write_dic:
        if func_name_lookup[key] in twotime_func_name_lst:
            with open(write_dic[key], 'w') as f:
                f.write('File created at {}. Author: Guang Shi\n'.format(datetime.date.today()))
                f.write('t0 t1 {}\n'.format(func_name_lookup[key]))
                np.savetxt(f, twotime_data_dic[twofunc_dic[key]], delimiter=' ')


# one-time quantity calculation
if onetime_flag == 1:
    onetime_data_dic = traj.cal_onetime(onefunc_dic.values(), screen_info=args.screen_info, **onetime_kwargs)
    onetime_output_name_lst = []
    for key in onefunc_dic.keys():
        onetime_output_name_lst.append(write_dic[key])
    for key in write_dic:
        if func_name_lookup[key] in onetime_func_name_lst:
            with open(write_dic[key], 'w') as f:
                if '.npy' in write_dic[key]:
                    np.save(f, onetime_data_dic[onefunc_dic[key]])
                else:
                    f.write('File created at {}. Author: Guang Shi\n'.format(datetime.date.today()))
                    np.savetxt(f, onetime_data_dic[onefunc_dic[key]])
