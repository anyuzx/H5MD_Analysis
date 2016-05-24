import numpy as np
import datetime
import pandas as pd

"""
Example:
ld = LammpsData()
ld.read('Chr5_145870001_157870001_SC.dat')
ld.write('Chr5_145870001_157870001_SC_duplicate.dat')

The above codes will write a data file same as its read in.
"""

"""
LammpsData.headers: dictionary where keys are atoms/atom types/...,
                                     value are the value cooresponding to keys
LammpsData.sections: dictionary where keys are Masses/Atoms/Bonds/...,
                                      value are the data list cooresponding to keys
"""

class LammpsData:
    def __init__(self):
        self.headers = {}
        self.sections = {}
        self.attribute = {}
        self.dataframe= {}

    def read(self, fp):
        headers = {} # store header information, like number of atoms, bonds, angles, box dimension, number of atom types ...
        sections = {} # store actual data for each header
        found = 0
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    headers['description'] = line[:-1]
                try:
                    for keyword in hkeywords:
                        if keyword in line:
                            if keyword == 'xlo xhi' or keyword == 'ylo yhi' or keyword == 'zlo zhi':
                                headers[keyword] = (float(line.split()[0]), float(line.split()[1]))
                            else:
                                headers[keyword] = int(line.split()[0])

                    for pair in skeywords:
                        keyword, length = pair[0], pair[1]
                        if keyword in line:
                            found = 1
                            start = i
                            length_temp = length
                            keyword_temp = keyword
                            sections[keyword_temp] = []
                            break

                    if found:
                        if i >= start + 2 and i <= start + 1 + headers[length_temp]:
                            sections[keyword_temp].append(line.split()[:])
                            continue
                        elif i == start + 3 + headers[length]:
                            found = 0
                            continue
                        else:
                            continue
                except:
                    continue

        self.headers = headers
        self.sections = sections

    def write(self, fp):
        with open(fp, 'w') as f:
            if 'description' not in self.headers or self.headers['description'] == '':
                f.write('Lammps Data File. Created at {}. Author: Guang Shi\n\n'.format(datetime.date.today()))
            else:
                f.write(self.headers['description']+'\n\n')

            for keyword in hkeywords:
                if keyword in self.headers:
                    value = self.headers[keyword]
                    if keyword == 'xlo xhi' or keyword == 'ylo yhi' or keyword == 'zlo zhi':
                        f.write('{:.6f} {:.6f} {}\n'.format(value[0], value[1], keyword))
                    else:
                        f.write('{} {}\n'.format(value, keyword))

            f.write('\n')

            for pair in skeywords:
                keyword = pair[0]
                if keyword in self.sections:
                    f.write('{}\n\n'.format(keyword))
                    lines = self.sections[keyword]
                    for line in lines:
                        f.write(' '.join('{}'.format(value) for value in line)+'\n')
                    f.write('\n')

    def GetDataFrame(self):
        xlo, xhi = self.headers['xlo xhi']
        ylo, yhi = self.headers['ylo yhi']
        zlo, zhi = self.headers['zlo zhi']
        self.dataframe['Box'] = pd.DataFrame(np.array([[xlo, xhi, ylo, yhi, zlo, zhi]]),
                                               columns=('xlo','xhi','ylo','yhi','zlo','zhi'))

        for keyword, value in self.sections.iteritems():
            if keyword == 'Atoms':
                if len(self.attribute) == 0:
                    self.dataframe[keyword] = pd.DataFrame(value)
                else:
                    attribute_lst = [None for i in range(len(self.attribute))]
                    for k, v in self.attribute.iteritems():
                        attribute_lst[v] = k
                    self.dataframe[keyword] = pd.DataFrame(value, columns=attribute_lst)
            else:
                self.dataframe[keyword] = pd.DataFrame(value, columns=dkeywords[keyword])

# ------------------------------------------------------------------------------
# define Lammps Data File keywords
hkeywords = ["atoms","ellipsoids","lines","triangles","bodies",
             "bonds","angles","dihedrals","impropers",
	     "atom types","bond types","angle types","dihedral types",
	     "improper types","xlo xhi","ylo yhi","zlo zhi","xy xz yz"]

skeywords = [["Masses","atom types"],
             ["Atoms","atoms"],["Ellipsoids","ellipsoids"],
             ["Lines","lines"],["Triangles","triangles"],["Bodies","bodies"],
             ["Bonds","bonds"],
	     ["Angles","angles"],["Dihedrals","dihedrals"],
	     ["Impropers","impropers"],["Velocities","atoms"],
             ["Pair Coeffs","atom types"],
	     ["Bond Coeffs","bond types"],["Angle Coeffs","angle types"],
	     ["Dihedral Coeffs","dihedral types"],
	     ["Improper Coeffs","improper types"],
             ["BondBond Coeffs","angle types"],
             ["BondAngle Coeffs","angle types"],
             ["MiddleBondTorsion Coeffs","dihedral types"],
             ["EndBondTorsion Coeffs","dihedral types"],
             ["AngleTorsion Coeffs","dihedral types"],
             ["AngleAngleTorsion Coeffs","dihedral types"],
             ["BondBond13 Coeffs","dihedral types"],
             ["AngleAngle Coeffs","improper types"],
             ["Molecules","atoms"]]

dkeywords = {'Masses': ('atom type', 'mass'),
            'Bonds': ('bond index', 'bond type', 'atom 1', 'atom 2'),
            'Angles': ('angle index', 'angle type', 'atom 1', 'atom 2', 'atom 3')}
# ------------------------------------------------------------------------------
