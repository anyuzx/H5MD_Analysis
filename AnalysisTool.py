import numpy as np
import LammpsH5MD
import msd
import isf
import sys
import yaml
import datetime

# ===================================================================
# define a dictionary for function name look up
func_name_dic = {
    'msd.g1': msd.g1,
    'msd.g2': msd.g2,
    'msd.g3': msd.g3,
    'isf':    isf.isf
}
# ===================================================================

# ===================================================================
# function read_parameter takes the yaml script as input,
# return
#   finname: path to the yaml format parameters file
#   func_lst: the function list which feed into LammpsH5MD.cal_correlate()
def read_parameter(script):
    func_dic = {}
    write_dic = {}
    for key, value in script.iteritems():
        if key == 'FILENAME':
            finname = value
        elif key == 'COMPUTE':
            for func in value:
                for func_name, info in func.iteritems():
                    if 'args' in info:
                        args = info['args']
                        func_dic[info['id']] = func_name_dic[func_name](**args)
                    else:
                        args = None
                        func_dic[info['id']] = func_name_dic[func_name]
        elif key == 'WRITE':
            for write in value:
                for write_name, info in write.iteritems():
                    write_dic[info['id']] = write_name

    return finname, script['ARGS'], write_dic, func_dic
# ===================================================================


# parameters file is written use YAML syntax
# load the parameters file using yaml
with open(sys.argv[1], 'r') as f:
    parameters = yaml.load(f)

finname, kwargs, write_dic, func_dic = read_parameter(parameters)

traj = LammpsH5MD.LammpsH5MD()    # create LammpsH5MD class
traj.load(finname)     # load the trajectory

# do the calculation
if 'variance' in kwargs and kwargs['variance']:
    mean, variance = traj.cal_correlate(func_dic.values(), **kwargs)
else:
    mean = traj.cal_correlate(func_dic.values(), **kwargs)

output_name_lst = []
for key in func_dic.keys():
    output_name_lst.append(write_dic[key])

if 'variance' in kwargs and kwargs['variance']:
    for index, quantity in enumerate(zip(mean,variance)):
        quantity = np.hstack((quantity[0], quantity[1][:,1:]))
        with open(output_name_lst[index], 'w') as f:
            f.write('File created at {}. Author: Guang Shi\n'.format(datetime.date.today()))
            f.write('# frames, mean, variance\n')
            np.savetxt(f, quantity, delimiter='    ')
else:
    for index, quantity in enumerate(mean):
        with open(output_name_lst[index], 'w') as f:
            f.write('File created at {}. Author: Guang Shi\n'.format(datetime.date.today()))
            f.write('# frames, mean\n')
            np.savetxt(f, quantity, delimiter='    ')
