''' 
    PYMTZ module implement I/O, manipulation and analysis of the 
    CCP4 mtz file format.
'''

import math, random, copy, os.path, os, tempfile, shutil, subprocess
import array as binarray

from scipy import zeros, array, nonzero, logical_not, sqrt, exp
from scipy import logical_and, cosh, log, sign, matrix, intersect1d
from scipy import unique, choose, setdiff1d, isinf, ones, pi, isfinite
from scipy import nanmin, nanmax, ceil, nanmean
from scipy.special import i0, erf
from scipy.integrate import quad
from numpy.random import randn, permutation, randint

import symoputils
from helper import readarray, list_to_float, list_to_int, list_to_strip

def read_mtz_file(mtzin):
    ''' Helper function that reads mtz-file and returns the mtz object.'''
    return mtz(mtzin)

def read_mtz_header(mtzin):
    ''' Helper function that reads mtz-file and returns MTZheader object. '''
    return MTZheader(mtzin)

__size_real__    = {4:'f', 8:'d'}
__size_complex__ = {4:'f', 8:'d'}
__size_int__     = {2:'i', 4:'l'}
__size_char__    = {1:'B', 2:'u'}

ALLOWED_MTZ_COLUMN_TYPES = {
'H': 'index h,k,l',
'J': 'intensity',
'F': 'structure amplitude, F',
'D': 'anomalous difference',
'Q': 'standard deviation of J,F,D or other (but see L and M below)',
'G': 'structure amplitude associated with one member of an hkl -h-k-l pair, F(+) or F(-)',
'L': 'standard deviation of a column of type G',
'K': 'intensity associated with one member of an hkl -h-k-l pair, I(+) or I(-)',
'M': 'standard deviation of a column of type K',
'E': 'structure amplitude divided by symmetry factor ("epsilon"). Normally scaled as well to give normalised structure factor',
'P': 'phase angle in degrees',
'W': 'weight (of some sort)',
'A': 'phase probability coefficients (Hendrickson/Lattman)',
'B': 'BATCH number',
'Y': 'M/ISYM, packed partial/reject flag and symmetry number',
'I': 'any other integer',
'R': 'any other real'}

def mu_centric(p):
    '''
        Returns the mu value for centric reflections in Lunin's maximum 
        likelihood formalism.
    '''
    mu = zeros(len(p))
    ind = nonzero(p>1.3)[0]
    pp = p[ind]
    s = pp*pp
    es = exp(-2.0*s)
    mu[ind] = (pp * ( 1.0 - 
                      2.0*es + 
                      (2.0 - 8.0*s)*es*es +
                      (-2.0 + 24.0*s - 48.0*s*s)*es*es*es + 
                      (2.0 - 48.0*s + 256.0*s*s - 341.3333*s*s*s)*es*es*es*es))
    ind = nonzero(logical_and(p>1.0, p<=1.3))[0]
    pp = p[ind]-1.0
    mu[ind] = sqrt(6.0*pp)*( 1.0 - 
                             0.55*pp + 
                             0.6944643*pp*pp - 
                             0.6409018*pp*pp*pp + 
                             0.6372297*pp*pp*pp*pp)
    return mu

def mu_acentric(p):
    '''
        Returns the mu value for acentric reflections in Lunin's maximum 
        likelihood formalism.
    '''
    mu = zeros(len(p))
    ind = nonzero(p>1.3)[0]
    s = 1.0/(p[ind]*p[ind])
    mu[ind] = p[ind]*( 1.0 - 
                       0.25*s - 
                       0.09375*s*s - 
                       0.0703125*s*s*s - 
                       0.06884766*s*s*s*s)
    ind = nonzero(logical_and(p>1.0, p<=1.3))[0]
    pp = p[ind]-1.0
    mu[ind] = sqrt(pp)*( 2.0 - 
                         0.8333333*pp + 
                         1.381944*pp*pp - 
                         1.231597*pp*pp*pp + 
                         1.126676*pp*pp*pp*pp)
    return mu

def nu_acentric(p):
    '''
        Returns the nu value for acentric reflections in Lunin's maximum 
        likelihood formalism.
    '''
    nu = zeros(len(p))
    ind = nonzero(p<=1.0)[0]
    p1 = p[ind]
    nu[ind] = 1 - p1*p1
    ind = nonzero(p>1)[0]
    p1 = p[ind]
    mu = mu_acentric(p1)
    nu[ind] = 2.0*(1.0-p1*p1+mu*mu)
    return nu

def nu_centric(p):
    '''
        Returns the nu value for centric reflections in Lunin's maximum 
        likelihood formalism.
    '''
    mu = mu_centric(p)
    return 1.0-p*p+mu*mu

def prob_acent(fobs, sigf, fcalc):
    ''' ML probability of Fo given the Fc in the absence of model errors for
        acentric reflection. '''
    return (fobs/sigf**2                         * 
            exp(-0.5*(fobs**2+fcalc**2)/sigf**2) *
            i0(fobs*fcalc/sigf**2))

def prob_cent(fobs, sigf, fcalc):
    ''' ML probability of Fo given the Fc in the absence of model errors for
        centric reflection. '''
    return (sqrt(2.0/pi)/sigf                    *
            exp(-0.5*(fobs**2+fcalc**2)/sigf**2) *
            cosh(fobs*fcalc/sigf**2))

def prob_norm(fobs, sigf, fcalc):
    ''' Gaussian probability of Fo given the Fc in the absence of model errors. 
    '''
    return (sqrt(0.5/pi)/sigf                    *
            exp(-0.5*(fobs-fcalc)**2/sigf**2))

def norm_fo(fcalc, sigf):
    ''' Expected Fo for a reflection given a perfect model and assuming
        that Fo's are normally distributed around Fc. ''' 
    a = fcalc / sigf
    return sigf * (a + exp(-0.5*a**2) / (1 + erf(a/sqrt(2))))

def norm_fofc(fcalc, sigf):
    ''' Expected Fo-Fc for a reflection given a perfect model and assuming
        that Fo's are normally distributed around Fc. ''' 
    a = fcalc / sigf
    return sigf * sqrt(2/pi) * (2-exp(-0.5*a**2)) / (1 + erf(a/sqrt(2)))

def cent_fo(fcalc, sigf):
    ''' Expected Fo for a centric reflection given a perfect model. ''' 
    a = fcalc / sigf
    return sigf * (a*erf(a/sqrt(2)) + sqrt(2/pi)*exp(-0.5*a**2))

def cent_fofc(fcalc, sigf):
    ''' Expected Fo-Fc for a centric reflection given a perfect model. ''' 
    a = fcalc / sigf
    return sigf * sqrt(2/pi) * (1 - exp(-0.5*a**2) + exp(-2*a**2) + 
                        a * sqrt(pi/2) * (2*erf(a*sqrt(2)) - erf(a/sqrt(2)) -1))

def acent_fo(fcalc, sigf, scope=10.0, alimit=10.0):
    ''' Expected Fo for an acentric reflection given a perfect model. ''' 
    a = fcalc / sigf
    if a>22.0:
        return sigf * (a + 0.5/a)
    if a>alimit:
        return sigf * (a + 0.5/a + 1e-6*(500.1-50.72*a+1.322*a**2))
    return (sigf * 
            quad(lambda x : x**2 * exp (-0.5*(x**2+a**2)) *i0(a*x), max(a-scope, 0.0), a+scope)[0] ) 

def acent_fofc(fcalc, sigf, scope=10.0, alimit=22.0):
    ''' Expected Fo-Fc for an acentric reflection given a perfect model. ''' 
    a = fcalc / sigf
    if a>alimit:
        return sigf * (1+exp(-2)/a**2)*sqrt(2/pi)
    return (sigf * 
            quad(lambda x : x * abs(x-a) * exp (-0.5*(x**2+a**2)) *i0(a*x), max(a-scope, 0.0), a+scope)[0] ) 

class MTZcell:
    '''
        The class that handles unit cell parameters.
    '''

    def __init__(self, unitcell):
        (self.cell_a, self.cell_b, self.cell_c, 
                        self.alpha, self.beta, self.gamma) = unitcell

    def set_cell(self, unitcell):
        ''' Set cell parameters with a tuple. '''
        (self.cell_a, self.cell_b, self.cell_c, 
                        self.alpha, self.beta, self.gamma) = unitcell

    def get_cell(self):
        ''' Get cell parameters as a tuple. '''
        return (self.cell_a, self.cell_b, self.cell_c, 
                                self.alpha, self.beta, self.gamma)

    def set_a(self, value):
        ''' Set the length of a. '''
        self.cell_a = value

    def get_a(self):
        ''' Get the length of a. '''
        return self.cell_a

    def set_b(self, value):
        ''' Set the length of b. '''
        self.cell_b = value

    def get_b(self):
        ''' Get the length of b. '''
        return self.cell_b

    def set_c(self, value):
        ''' Set the length of c. '''
        self.cell_c = value

    def get_c(self):
        ''' Get the length of c. '''
        return self.cell_c

    def set_alpha(self, value):
        ''' Set the alpha angle. ''' 
        self.alpha = value

    def get_alpha(self):
        ''' Get the alpha angle. ''' 
        return self.alpha

    def set_beta(self, value):
        ''' Set the beta angle. ''' 
        self.beta = value

    def get_beta(self):
        ''' Get the beta angle. ''' 
        return self.beta

    def set_gamma(self, value):
        ''' Set the gamma angle. ''' 
        self.gamma = value

    def get_gamma(self):
        ''' Get the gamma angle. ''' 
        return self.gamma

    def get_scoeffs(self):
        ''' Returns coefficients for the conversion of Miller indices into the
            scattering vector. '''
        (alpha, beta, gamma) = (    math.radians(self.alpha), 
                                      math.radians(self.beta), 
                                        math.radians(self.gamma))
        (sin_a, sin_b, sin_c) = (   math.sin(alpha), 
                                      math.sin(beta), 
                                        math.sin(gamma))
        (cos_a, cos_b, cos_c) = (   math.cos(alpha), 
                                      math.cos(beta), 
                                        math.cos(gamma))
        v_factor = 1.0 - cos_a**2 - cos_b**2 -cos_c**2 +2.0*cos_a*cos_b*cos_c
        return ((sin_a/self.cell_a)**2/v_factor, 
                (sin_b/self.cell_b)**2/v_factor, 
                (sin_c/self.cell_c)**2/v_factor, 
                2.0*(cos_a*cos_b-cos_c)/self.cell_a/self.cell_b/v_factor,
                2.0*(cos_a*cos_c-cos_b)/self.cell_a/self.cell_c/v_factor,
                2.0*(cos_b*cos_c-cos_a)/self.cell_b/self.cell_c/v_factor)

    def get_volume(self):
        ''' Return the unit cell volume. '''
        abc = self.cell_a * self.cell_b * self.cell_c
        (alpha, beta, gamma) = (math.radians(self.alpha), 
                                  math.radians(self.beta), 
                                    math.radians(self.gamma))
        (cos_a, cos_b, cos_c) = (   math.cos(alpha), 
                                      math.cos(beta), 
                                        math.cos(gamma))
        return abc*math.sqrt(1.0 - cos_a**2 - cos_b**2 -cos_c**2 
                                                    + 2.0*cos_a*cos_b*cos_c)

class SymmInfo:
    ''' The class holds symmetry information.  Should be instantiated with a
        line that lists the number of symmetry operators (nsym), the number
        of primitive symmetry operators (nprimitive), the lattice type 
        (lattype, i.e. P/I/C/H/R), the space group number (sgnumber), the
        space group name (sgname, in single quotes, e.g. 'P 21 21 21'),
        and the point group name (pgname, e.g PG222).'''

    def __init__(self, line):
        self.nsym = int(line.split()[0])
        self.nprimitive = int(line.split()[1])
        self.lattype = line.split()[2]
        self.sgnumber = int(line.split()[3])
        self.sgname = line.split("'")[1]
        self.pgname = line.split("'")[2].strip()

    def set_nsym(self, value):
        ''' Set the number of symmetry operators. '''
        self.nsym = value

    def get_nsym(self):
        ''' Get the number of symmetry operators. '''
        return self.nsym

    def set_nprim(self, value):
        ''' Set the number of primitive operators. '''
        self.nprimitive = value

    def get_nprim(self):
        ''' Get the number of primitive operators. '''
        return self.nprimitive

    def set_lattice_type(self, value):
        ''' Set the lattice type. '''
        self.lattype = value

    def get_lattice_type(self):
        ''' Get the lattice type. '''
        return self.lattype

    def set_spacegroup_number(self, value):
        ''' Set the space group number. '''
        self.sgnumber = value

    def get_spacegroup_number(self):
        ''' Get the space group number. '''
        return self.sgnumber

    def set_spacegroup_name(self, value):
        ''' Set the space group name. '''
        self.sgname = value

    def get_spacegroup_name(self):
        ''' Get the space group name. '''
        return self.sgname

    def set_pointgroup_name(self, value):
        ''' Set the point group name. '''
        self.pgname = value

    def get_pointgroup_name(self):
        ''' Get the point group name. '''
        return self.pgname

class SymmetryOperator:
    ''' The symmetry operator class. '''

    def __init__(self, ops):
        (self.op_x, self.op_y, self.op_z) = ops

    def set_x(self, value):
        ''' Set the X operator. '''
        self.op_x = value

    def get_x(self):
        ''' Returns the X operator. '''
        return self.op_x

    def set_y(self, value):
        ''' Set the Y operator. '''
        self.op_y = value

    def get_y(self):
        ''' Returns the Y operator. '''
        return self.op_y

    def set_z(self, value):
        ''' Set the Z operator. '''
        self.op_z = value

    def get_z(self):
        ''' Returns the Z operator. '''
        return self.op_z

    def get_line(self):
        ''' Returns the standard string form of the symmetry operator.'''
        return self.op_x+',  '+self.op_y+',  '+self.op_z

    def get_operator(self):
        ''' Return the lambda-form of the symmetry operator. '''
        return eval('lambda x,y,z: (' + 
                      self.op_x.lower().replace('/','./') + ',' + 
                        self.op_y.lower().replace('/','./') + ',' + 
                          self.op_z.lower().replace('/','./') + ')')

class MTZcolumn:
    ''' The class that defines a column in MTZ file (not the data itself). '''

    def __init__(self, items):
        (self.label, self.type) = items[:2]
        (self.min, self.max) = list_to_float(items[2:4])
        self.dset_id = int(items[4])

    def set_label(self, value):
        ''' Sets the column label. '''
        self.label = value

    def get_label(self):
        ''' Returns the column label. '''
        return self.label

    def set_type(self, value):
        ''' Sets the column type. '''
        self.type = value

    def get_type(self):
        ''' Returns the column type. '''
        return self.type

    def set_min(self, value):
        ''' Sets the minimum value for the column. '''
        self.min = value

    def get_min(self):
        ''' Returns the minimum value of the column.  This is the stored value
            and is not dynamically re-evaluated if column data change. '''
        return self.min

    def set_max(self, value):
        ''' Sets the maximum value for the column. '''
        self.max = value

    def get_max(self):
        ''' Returns the maximum value of the column.  This is the stored value
            and is not dynamically re-evaluated if column data change. '''
        return self.max

    def set_min_max(self, minmax):
        ''' Sets both minimum and maximum value of the column with a tuple. '''
        (self.min, self.max) = minmax

    def get_min_max(self):
        ''' Returns the tuple of both minimum and maximum value of the column.
            These are the stored values and are not dynamically re-evaluated 
            if column data change. '''
        return (self.min, self.max)

    def set_dataset_id(self, value):
        ''' Sets the column dataset ID (generally a number). '''
        self.dset_id = value

    def get_dataset_id(self):
        ''' Returns the column dataset ID. '''
        return self.dset_id

class MTZoriblock:
    ''' The class reads the general data block.  This has to do with
        multi-record files. '''

    def __init__(self, fin):
        self.nums = readarray(fin, 'I', 3)
        self.intbuf = readarray(fin, 'I', self.nums[1])
        self.floatbuf = readarray(fin, 'f', self.nums[2])
        
    def get_nwords(self):
        ''' Returns the number of words in the block. '''
        return self.nums[0]

    def get_nintgr(self):
        ''' Returns the number of integer words in the block. '''
        return self.nums[1]

    def get_nreals(self):
        ''' Return the number of float words in the block. '''
        return self.nums[2]


class mtz:
    ''' The mtz class to manipulate the content if an mtz file. '''

    def __init__(self, fname):
        fin = open(fname,'rb')
        mtzstamp = readarray(fin, 'B', 4)
        if mtzstamp.tobytes().decode() != 'MTZ ':
            raise TypeError(fname+' is not an MTZ-file')
        headerposition = readarray(fin)
        machinestamp =  readarray(fin, 'B', 4)
        self.__initialize()
        fin.seek((headerposition[0]-1)*4)
        record_flag = True
        while record_flag:
            record = readarray(fin, 'B', 80)
            record_flag = self.assign_record(record.tobytes().decode())
            if record_flag == 'END':
                historyposition = fin.tell()
                record_flag = False
        nhis = int((fin.tell()-historyposition)/80)-1
        f_batches = False
        fin.seek(historyposition)
        for i in range(nhis):
            record = readarray(fin, 'B', 80)
            hisline = record.tostring()
            if hisline[:7] == 'MTZBATS':
                f_batches = True
                break
            else:
                self.history.append(record.tostring())
        if f_batches:
            while True:
                record = readarray(fin, 'B', 80)
                keys = record.tostring().split()
                if keys[0] == 'BH':
                    curb = int(keys[1])
                    self.batchlist[curb] = [keys[2:]]
                elif keys[0] == 'TITLE':
                    self.batchlist[curb].append(' '.join(keys[1:]))
                    self.batchlist[curb].append(MTZoriblock(fin))
                    break
        fin.seek(80)
        for i in range(self.nref):
            record = readarray(fin, 'f', self.ncol)
            self.reflections.append(record.tolist())
        fin.close()
        self.hind, self.kind, self.lind = (self.GetColumnIndex('H'), 
                                             self.GetColumnIndex('K'), 
                                               self.GetColumnIndex('L'))
        self.hsh = None

    def __initialize(self):
        ''' Private method that initialize classes data. '''
        (self.mtzversion, self.mtztitle) = ['']*2
        (self.ncol, self.nref, self.nbat, self.ndat) = [0]*4
        (self.globalcell, self.syminfo, self.lowres, 
                                         self.highres, self.missflag) = [None]*5
        self.sortorder = [0]*5
        (self.symm, self.columns, self.history, 
                                            self.reflections) = [], [], [], []
        (self.projects, self.crystals, self.datasets, 
                                self.dcell, self.dwavel) = {}, {}, {}, {}, {}

    def __mtz_machine_stamp(self):
        '''
            This private method returns the machine stamp as defined in 
            MTZ format.
        '''
        stamp = binarray.array('B')
        stamp.append(binarray.array('f').itemsize*16 + 
                     binarray.array('f').itemsize)
        stamp.append(binarray.array('i').itemsize*16 + 
                     binarray.array('B').itemsize)
        stamp.append(0)
        stamp.append(0)
        return stamp

    def assign_batch_record(self, line):
        ''' Assigns the batch record. Needs fixing'''
        keys = line.split()
        if keys[0] == 'BH':
            self.batchlist[int(b)] = None

    def assign_record(self, line):
        ''' Assigns an MTZ record based on the content of the string. '''
        key = line.split()[0]
        if key == 'VERS':
            self.SetMTZVersion(line.split(key)[1].strip().ljust(10))
        elif key == 'TITLE':
            self.SetMTZTitle(line.split(key)[1].strip().ljust(70))
        elif key == 'NCOL':
            (ncol, nref, nbat) = line.split(key)[1].split()
            self.SetColumnNumber(int(ncol))
            self.SetReflectionNumber(int(nref))
            self.SetBatchNumber(int(nbat))
#            if self.nbat>0:
#                return False
        elif key == 'CELL':
            self.SetGlobalCell(line.split(key)[1].split())
        elif key == 'SORT':
            self.SetSortOrder(line.split(key)[1].split())
        elif key == 'SYMINF':
            self.SetSymInfo(line.split(key)[1])
        elif key == 'SYMM':
            self.AddSymmetryOperator(list_to_strip(line.split(key)[1].split(',')))
        elif key == 'RESO':
            self.SetResolution(list_to_float(line.split(key)[1].split()))
        elif key == 'VALM':
            self.SetMissingFlag(line.split(key)[1].strip())
        elif key == 'COLUMN' or key == 'COL':
            self.AddColumn(line[len(key):].split())
        elif key == 'NDIF':
            self.SetDatasetNumber(int(line[len(key):].split()[0]))
        elif key == 'PROJECT':
            self.AddProjectName(line[len(key):].split())
        elif key == 'CRYSTAL':
            self.AddCrystalName(line[len(key):].split())
        elif key == 'DATASET':
            self.AddDatasetName(line[len(key):].split())
        elif key == 'DCELL':
            self.AddDatasetCell(line[len(key):].split())
        elif key == 'DWAVEL':
            self.AddDatasetWavelength(line[len(key):].split())
        elif key == 'BATCH':
            self.AddBatches(line[len(key):].split())
        elif key == 'END':
            return key
        elif key == 'MTZENDOFHEADERS':
            return False
        return True

    def append_column(self, values, c_label, c_type, dset_id):
        ''' This method should be used (with caution) to insert extra columns
            into the mtz file.  Provide values for the column data, must be the
            same size as all the other columns.  c_label is the label for the 
            new column which must be new, c_type is the column type (see 
            standard column types in mtzlib).  dset_id parameter allows to 
            attach the new data column to a specific dataset, make sure it 
            exists already.'''
        items = []
        if c_label in self.GetLabels():
            raise ValueError(c_label+' column already present')
        else:
            items.append(c_label)
        if c_type in ALLOWED_MTZ_COLUMN_TYPES:
            items.append(c_type)
        else:
            raise ValueError(c_type+' is not an allowed column type')
        items += [float(min(values)), float(max(values))]
        if dset_id in self.datasets:
            items.append(dset_id)
        else:
            raise ValueError('No dataset '+str(dset_id))
        if len(values) != self.nref:
            raise ValueError('Incoming column length mismatch: ' + 
                                        str(len(values))+'/'+str(self.nref))
        self.AddColumn(items)
        reflections = array(self.reflections).T.tolist()
        reflections.append(values)
        self.reflections = array(reflections).T.tolist()
        self.ncol += 1

    def duplicate_column(self, label, dup_label):
        ''' Make an exact copy of a column, giving it a different label. '''
        if label not in self.GetLabels():
            raise KeyError('Column with the '+label+' not found')
        column = self.GetColumn(label)
        values = self.GetReflectionColumn(label)
        coltype = column.get_type()
        dset_id = column.get_dataset_id()
        self.append_column(values, dup_label, coltype, dset_id)

    def acentrics(self):
        ''' Return the vector of acentric flags (boolean). '''
        hkl = self.GetHKLArray()
        sgn = self.GetSymInfo().get_spacegroup_number()
        rots = symoputils.symop_uniq_rots(sgn)
        acent_flag = ones(self.GetReflectionNumber())
        for rot in rots:
            hklr = array(rot*hkl.T).T.round()
            acent_flag *= (hklr+hkl).any(1).astype(int)
        return acent_flag.astype(bool)

    def perfect_fofc(self, labels=['SIGFP','FC_ALL','FP'], mode='rice', est_fobs=True):
        ''' Calculates the predicted values of the Fo's and the |Fo-Fc|
            difference assuming that a perfect model is available.  Labels
            provide the sigmaF and Fc columns.  Three calculation modes are 
            available: 'trivial', 'normal', and 'rice'.  The latter will be
            fairly slow since no analytical solution exists for acentric
            reflections and directed integration is done for each reflection.'''
        sigfp, fcalc = self.GetReflectionColumns(labels[:2])
        if not est_fobs:
            fp = self.GetReflectionColumn(labels[2])
        if mode.lower()[:4] == 'norm':
            if est_fobs:
                fobs = norm_fo(fcalc, sigfp)
            else:
                fobs = fp
            dfs = norm_fofc(fcalc, sigfp)
        elif mode.lower()[:4] == 'triv':
            if est_fobs:
                fobs = fcalc
            else:
                fobs = fp
            dfs = sqrt(2/pi) * sigfp
        elif mode.lower()[:4] == 'rice':
            acflags = self.acentrics()
            dfs = []
            for (i, acflag) in enumerate(acflags):
                if isfinite(sigfp[i]):
                    if acflag:
                        dfs.append(acent_fofc(fcalc[i], sigfp[i]))
                    else:
                        dfs.append(cent_fofc(fcalc[i], sigfp[i]))
                else:
                    dfs.append(float('nan'))
            if est_fobs:
                fobs = []
                for (i, acflag) in enumerate(acflags):
                    if isfinite(sigfp[i]):
                        if acflag:
                            fobs.append(acent_fo(fcalc[i], sigfp[i]))
                        else:
                            fobs.append(cent_fo(fcalc[i], sigfp[i]))
                    else:
                        fobs.append(float('nan'))
            else:
                fobs = fp
        return array([fobs,dfs])

    def perfect_rvalue(self, labels=['SIGFP','FC_ALL', 'FP'], mode='rice', est_fobs=True):
        ''' Calculates the predicted R-value assuming that a perfect model is 
            available.  See perfect_fofc() method for details. '''
        fobs,dfs = self.perfect_fofc(labels=labels, mode=mode, est_fobs=est_fobs)
        return nanmean(dfs)/nanmean(fobs)

    def SetMTZVersion(self, line):
        ''' Sets the MTZ version with a line. '''
        self.mtzversion = line

    def GetMTZVersion(self):
        ''' Returns the line describing the MTZ version. '''
        return self.mtzversion

    def SetMTZTitle(self, line):
        ''' Stes the data file title with a ine. '''
        self.mtztitle = line
            
    def GetMTZTitle(self):
        ''' Return the data file title. '''
        return self.mtztitle

    def SetColumnNumber(self, ncol):
        ''' Sets the total number of columns.  Do not use unless the actual
            number of columns is being actually changed. '''
        self.ncol = ncol

    def GetColumnNumber(self):
        ''' Returns the total number of columns in the data file. '''
        return self.ncol

    def SetReflectionNumber(self, nref):
        ''' Sets the total number of reflection.  Use carefully. '''
        self.nref = nref

    def GetReflectionNumber(self):
        ''' Returns the number of reflections in the file. '''
        return self.nref

    def SetBatchNumber(self, nbat):
        ''' Sets the number of reflection batches.  Use carefully. '''
        self.nbat = nbat
        self.batchlist = {}

    def AddBatches(self, blist):
        ''' Add batches using a list.  This method is doing nothing right now
            but setting an internal "batch list".  Do not use. '''
        for b in blist:
            self.batchlist[int(b)] = None

    def GetBatchNumber(self):
        ''' Returns the number of batches in the data file. '''
        return self.nbat

    def SetGlobalCell(self, unitcell):
        ''' Sets the global unit cell.  
                Global cell concept is outdated, right? '''
        self.globalcell = MTZcell(list_to_float(unitcell))

    def GetGlobalCell(self):
        ''' Returns the global unit cell. '''
        return self.globalcell

    def SetSortOrder(self, sorder):
        ''' Sets the miller indices sort order.  This method does not actually
            re-order reflections. '''
        self.sortorder = list_to_int(sorder)

    def GetSortOrder(self):
        ''' Returns the miller indices sorting order. '''
        return self.sortorder

    def SetSymInfo(self, line):
        ''' Sets the symmetry info object with the line.  See SymmInfo for 
            more details. '''
        self.syminfo = SymmInfo(line)

    def GetSymInfo(self):
        ''' Returns the symmetry info object. '''
        return self.syminfo

    def AddSymmetryOperator(self, op):
        ''' Adds a symmetry operator to the list. '''
        self.symm.append(SymmetryOperator(op))

    def ClearSymmetryOperators(self):
        ''' Deletes all the currently stored symmetry operators. '''
        self.symm = []

    def GetSymmetryOperators(self):
        ''' Returns the stored list of symmetry operators. '''
        return self.symm

    def SetResolution(self, res):
        ''' Sets the resolution limits.  The passed value should be amenable
            to extracting a maximum and minimum values. '''
        self.lowres = min(res)
        self.highres = max(res)

    def SetLowResolution(self, res):
        ''' Sets the low resolution limit. '''
        self.lowres = res

    def SetHighResolution(self, res):
        ''' Sets the high resolution limit. '''
        self.highres = res

    def GetLowResolution(self):
        ''' Returns the low resolution limit. '''
        return self.lowres

    def GetHighResolution(self):
        ''' Returns the high resolution limit. '''
        return self.highres

    def GetLowD(self):
        '''Returns the low resolution scattering vector length. '''
        return pow(self.GetLowResolution(), -0.5)

    def GetHighD(self):
        '''Returns the high resolution scattering vector length. '''
        return pow(self.GetHighResolution(), -0.5)

    def SetMissingFlag(self, value):
        ''' Sets the missing value flag. '''
        self.missflag = value

    def GetMissingFlag(self):
        ''' Returns the missing value flag. '''
        return self.missflag

    def AddColumn(self, items):
        ''' Add a column using the set of items (i.e. lines from an mtz file
            header. '''
        self.columns.append(MTZcolumn(items))

    def GetColumns(self):
        ''' Returns the list of column objects. '''
        return self.columns

    def SetDatasetNumber(self, value):
        ''' Sets the number of datasets present in the mtz file. '''
        self.ndat = value

    def GetDatasetNumber(self):
        ''' Returns the number of datasets present in the mtz file. '''
        return self.ndat

    def AddProjectName(self, items):
        ''' Add a project name to the list of datasets. '''
        self.projects[int(items[0])] = items[1]

    def SetProjectName(self, dset_id, value):
        ''' Sets the project name for a dataset. '''
        self.projects[dset_id] = value

    def GetProjectName(self, dset_id):
        ''' Returns the project name of a dataset. '''
        return self.projects[dset_id]

    def AddCrystalName(self, items):
        ''' Add a crystal name to the list of datasets. '''
        self.crystals[int(items[0])] = items[1]

    def SetCrystalName(self, dset_id, value):
        ''' Sets the crystal name for a dataset. '''
        self.crystals[dset_id] = value

    def GetCrystalName(self, dset_id):
        ''' Returns the crystal name of a dataset. '''
        return self.crystals[dset_id]

    def AddDatasetName(self, items):
        ''' Adds a dataset name for the list. '''
        self.datasets[int(items[0])] = items[1]

    def SetDatasetName(self, dset_id, value):
        ''' Sets the dataset name. '''
        self.datasets[dset_id] = value

    def GetDatasetName(self, dset_id):
        ''' Returns the dataset name. '''
        return self.datasets[dset_id]

    def AddDatasetCell(self, items):
        ''' Adds the unit cell for a dataset. ''' 
        self.dcell[int(items[0])] = MTZcell(list_to_float(items[1:]))

    def SetDatasetCell(self, dset_id, unitcell):
        ''' Sets the unit cell for a dataset. ''' 
        self.dcell[dset_id] = MTZcell(unitcell)

    def GetDatasetCell(self, dset_id):
        ''' Returns the unit cell for a dataset. ''' 
        return self.dcell[dset_id]

    def GetDatasetVolume(self, dset_id):
        ''' Returns the volume of the unit cell for a dataset. '''
        return self.dcell[dset_id].get_volume()

    def AddDatasetWavelength(self, items):
        ''' Adds the wavelength value to the list of the datasets. '''
        self.dwavel[int(items[0])] = float(items[1])

    def SetDatasetWavelength(self, dset_id, value):
        self.dwavel[dset_id] = value

    def GetDatasetWavelength(self, dset_id):
        return self.dwavel[dset_id]

    def AddHistoryRecord(self, line):
        self.history.append(line.ljust(80)[:80])

    def GetHistory(self):
        return self.history

    def ClearHistory(self):
        self.history = []

    def GetAllReflections(self):
        return self.reflections

    def SetAllReflections(self, value):
        self.reflections = value

    def GetSingleReflection(self, refindex):
        if refindex < self.nref:
            return self.reflections[refindex]
        else:
            return None

    def SetSingleReflection(self, refindex, value):
        if refindex < self.nref:
            self.reflections[refindex] = value

    def GetColumnIndex(self, label):
        for (i, column) in enumerate(self.columns):
            if column.get_label() == label:
                return i
        return None

    def GetColumn(self, label):
        for column in self.columns:
            if column.get_label() == label:
                return column
        return None

    def GetSingleReflectionHKL(self, hkl):
        for reflection in self.reflections:
            if reflection[self.hind] == hkl[0]:
                if reflection[self.kind] == hkl[1]:
                    if reflection[self.lind] == hkl[2]:
                        return reflection
        return None

    def GetResolutionColumn(self, dset_id=-1, force=False):
        ''' Returns the vector of d-spacing for individual reflections.
            Deafults to the first dataset (normally HKL unless the mtz file
            was really weird).  It stores the result in the class first time
            the method is invoked, use force=True to force recalculation.
            Notice that it is not reset/erased internally when relevant
            dataset information (i.e. cell parameters, wavelength) changes.
            Any method that does such modification should either erase
            rescol or force recalculation.'''
        if 'rescol' not in dir(self) or force:
            dset_id = self.__dset_id_verify_(dset_id)
            self.rescol = 1/sqrt(array(self.GetS2(dset_id)))
        return self.rescol

    def GetS2(self, dset_id):
        (s11, s22, s33, s12, s13, s23) = self.GetDatasetCell(dset_id).get_scoeffs()
        return [s11*h**2+s22*k**2+s33*l**2+s12*h*k+s13*h*l+s23*k*l for h,k,l in zip(*[self.GetReflectionColumn('H'),self.GetReflectionColumn('K'),self.GetReflectionColumn('L')])]

    def GetReflectionColumn(self, label):
        ''' Return the vector of column values as a tuple.'''
        colind = self.GetColumnIndex(label)
        if colind == None:
            return None
        else:
            return list(zip(*self.reflections))[colind]
#            column = []
#            for reflection in self.reflections:
#                column.append(reflection[colind])
#            return column

    def GetColumnDictionary(self):
        ''' Returns the dictionary of the available reflection columns.  The
            keys are the column labels while values are arrays of data. '''
        x = array(self.reflections).T
        columns = {}
        for (i,label) in enumerate(self.GetLabels()):
            columns[label] = x[i]
        return columns
            
    def nancorr(self, label1, label2):
        ''' Returns the correlation coefficient between the two columns.''' 
        col1, col2 = self.GetColumnIndex(label1), self.GetColumnIndex(label2)
        if col1==None or col2==None:
            return None
        x = array(self.reflections).T
        return self.__nancorr_(x[col1], x[col2])

    def __nancorr_(self, fi,fj):
        fifj = fi*fj
        ind = isfinite(fifj)
        return (fifj[ind].mean()-fi[ind].mean()*fj[ind]).mean()/fi[ind].std()/fj[ind].std()


    def nancorrd(self, label1, label2, num=10, dset_id=-1):
        ''' Returns the correlation coefficient between the two columns as a
            function of resolution.''' 
        col1, col2 = self.GetColumnIndex(label1), self.GetColumnIndex(label2)
        if col1==None or col2==None:
            return None
        d = self.GetResolutionColumn()
        splitter = self.shell_splitter(num, dset_id)
        x = array(self.reflections).T
        retval = [];
        for ind in splitter:
            retval.append([d[ind].mean(), self.__nancorr_(x[col1][ind], x[col2][ind])])
        return array(retval)


    def GetReflectionColumns(self, labels):
        ''' Returns the list where each element is an array of values from the
            columns that correspond to the labels from the argument.''' 
        return array(self.reflections).T[nonzero([(t in labels) for t in self.GetLabels()])[0]].T

    def GetHKLArray(self):
        return array(self.GetReflectionColumns('HKL')).T

# --- All things hash

    def GetHKLhash(self, base=1000, hkl=None):
        ''' Return the array of values that uniquely identify each reflection.
            The hash is constructed as
             
                base*(abs(L)+base*(abs(K)+base*(abs(H)+base*signature)).  
            
            The last item, signature, is designed to distinguish the sign of
            corresponding Miller indices, and is constructed like this
            
                signature = 1 + s_h + 2*s_k + 4*s_l
                
            where s_x is int((sign(x)+1)/2).  For example,
            with default value of the base=1000, reflection (17,2,11) will get 
            the hash value of 4017002011.  Using a different base may be needed
            only if the maximum miller index is >999, which is highly unlikely 
            scenario. This method is useful for sorting and finding equivalent 
            reflections in two datasets.
            
            If hkl is provided, the hash will be calculated for the given set
            of reflections.  The array must contain miller indices arranged
            either row- or column-wise.  The method assumes that if a 3x3
            array is provided, the firs dimension runs is the reflection, 
            i.e. the first column contains h, second k and the third l.'''
        if hkl == None:
            hkl = array(self.GetReflectionColumns('HKL')).T
        try:
            if hkl.shape[1] != 3:
                hkl = hkl.T
        except IndexError:
            if hkl.ndim != 1:
                raise
            hkl = array([hkl])
        sgntr = 1+((((sign(hkl)+1)/2).astype(int))*[1, 2, 4]).sum(1)
        return (abs(hkl.astype(int))*[10**6, 1000, 1]).sum(1)+10**9*sgntr

    def hklash(self, base=1000):
        ''' Generates hkl hashes and stores them as internal value.  See 
            GetHKLhash() method for further details. '''
        hkl = array(self.GetReflectionColumns('HKL')).T
        sgntr = 1+((((sign(hkl)+1)/2).astype(int))*[1, 2, 4]).sum(1)
        self.hsh = (abs(hkl.astype(int))*[10**6, 1000, 1]).sum(1)+10**9*sgntr

    def hash2hkl(self, hsh, base=1000):
        ''' Converts reflection hash to miller indices. '''
        sgntr = hsh / base**3 - 1
        return (choose(array([sgntr % 2, sgntr/2 % 2, sgntr/4]), [-1, 1]) * 
                       array([hsh/base**2 % base, hsh/base % base, hsh % base]))

    def hklindex(self, hkl, base=1000):
        listik = []
        for hsh in self.GetHKLhash(base=base, hkl=hkl):
            listik.append(nonzero(self.hsh==hsh)[0][0])
        return listik

    def hashindex(self, hshs):
        listik = []
        for hsh in hshs:
            listik.append(nonzero(self.hsh==hsh)[0][0])
        return listik

# ---

    def GetRedundancies(self, nomerge):
        if self.hsh == None:
            self.hklash()
        if nomerge.hsh == None:
            nomerge.hklash()
        nomerge_hsh = nomerge.hsh.copy()
        nomerge_hsh.sort()
        ind1 = self.hsh.argsort()
        ind2 = ind1.argsort()
        ind = array(nonzero(nomerge_hsh[1:] - 
                    nomerge_hsh[:-1])[0].tolist() + 
                        [len(nomerge_hsh)-1])
        seq = array([-1]+ind.tolist())
        freq = seq[1:]-seq[:-1]
        if (self.hsh[ind1] == nomerge_hsh[ind]).any():
            return freq[ind2]
        else:
            raise ValueError('Problem matching hkl hashes. ' + 
                    'Make sure that the correct unmerged file is used.  ' + 
                             'Otherwise, report the bug.')

    def GenerateRedundancyColumn(self, nomerge, 
                                            label='REDUNDANCY', dset_id=None):
        if not dset_id:
            dset_id = max(mtz1.datasets)
        self.append_column(self.GetRedundancies(nomerge), label, 'I', dset_id)

    def GetMapArray(self, f, phi):
        return array(self.GetReflectionColumns(['H', 'K', 'L', f, phi])).T

    def GetResolutionAverages(self, label, nshells=10):
        f = array(self.GetReflectionColumn(label))
        sh = array(self.GetShellColumn(nshells))
        d = array(self.GetResolutionColumn())
        d_sh, f_sh = [], []
        for i in range(1, nshells+1):
            ind = nonzero(logical_not(sh-i))
            d_sh.append(d[ind].mean())
            f_sh.append(nanmean(f[ind]))
        return (array(d_sh), array(f_sh))

    def GetVectorResolutionAverages(self, vector, nshells=10):
        f = array(vector)
        sh = array(self.GetShellColumn(nshells))
        d = array(self.GetResolutionColumn())
        d_sh, f_sh = [], []
        for i in range(1, nshells+1):
            ind = nonzero(logical_not(sh-i))
            d_sh.append(d[ind].mean())
            f_sh.append(nanmean(f[ind]))
        return (array(d_sh), array(f_sh))

    def SetReflectionColumn(self, label, values):
        colind = self.GetColumnIndex(label)
        if colind == None:
            raise KeyError('No such column: |'+label+'|')
        elif len(values) != self.nref:
            raise ValueError('Incoming column length mismatch: ' + 
                                        str(len(values))+'/'+str(self.nref))
        else:
            for (i, value) in enumerate(values):
                self.reflections[i][colind] = value
        self.ResetMinMax(label)

    def GetZscore(self, f='F', fc='FC', sigf='SIGF'):
        col_f, col_fc, col_sigf = (self.GetColumnIndex(f), 
                                    self.GetColumnIndex(fc), 
                                     self.GetColumnIndex(sigf))
        if col_f == None or col_fc == None or col_sigf == None:
            return None
        else:
            z = []
            for reflection in self.reflections:
                z.append((reflection[col_f]-reflection[col_fc]) / 
                                                        reflection[col_sigf])
            return z

    def GetResolutionShells(self, num=10, dset_id=-1):
        sr = sorted(self.GetResolutionColumn(dset_id), reverse=True)
        shells = []
        for i in range(num):
            shells.append(sr[int(self.nref*i/num)])
        shells.append(sr[-1])
        return shells

    def GetShellColumn(self, num=10, dset_id=-1):
        if dset_id < 0:
            dset_id = min(self.dcell.keys())
        shells = self.GetResolutionShells(num, dset_id) if type(num)==int else num
        d = self.GetResolutionColumn(dset_id)
        shid = zeros(self.GetReflectionNumber())
        for edge in shells[1:-1]:
            shid += (d<edge).astype(int)
        return shid

    def ShellSplit(self, column, num=10, dset_id=-1):
        if dset_id < 0:
            dset_id = min(self.dcell.keys())
        if type(column) == str:
            column = self.GetReflectionColumn(column)
        if column == None:
            return None
        splits = []
        for i in range(num):
            splits.append([])
        shids = self.GetShellColumn(num, dset_id)
        for (i, shid) in enumerate(shids):
            splits[shid-1].append(column[i])
        return splits

    def GetLabels(self):
        ''' Return the list of column labels.'''
        labels = []
        for column in self.columns:
            labels.append(column.get_label())
        return labels

    def GetTypes(self):
        types = []
        for column in self.columns:
            types.append(column.get_type())
        return types

    def SetColumnType(self, label, tp):
        self.GetColumn(label).set_type(tp) 

    def RenameColumn(self, label_pair):
        (label1, label2) = label_pair
        coli1 = self.GetColumnIndex(label1)
        coli2 = self.GetColumnIndex(label2)
        if coli1 != None:
            self.columns[coli1].set_label(label2)
            return coli1
        elif coli2 != None:
            self.columns[coli2].set_label(label1)
            return coli2
        else:
            return None

    def RenameColumns(self, label_pairs):
        colis = []
        for label_pair in label_pairs:
            colis.append(self.RenameColumn(label_pair))
        return colis

    def LabelFix(self, styles):
        ''' Convert labels from one standard to another.  Available style 
            conversions are (self-explanatory):
                phenix_refmac
                scala_refmac
        '''
        if styles == 'phenix_refmac':
            self.RenameColumns([('FOBS', 'FP'), 
                                ('SIGFOBS', 'SIGFP'), 
                                ('R_FREE_FLAGS', 'FREE'), 
                                ('FMODEL', 'FC_ALL'), 
                                ('PHIFMODEL', 'PHIC_ALL'), 
                                ('FCALC', 'FC'), 
                                ('PHIFCALC', 'PHIC'), 
                                ('2FOFCWT', 'FWT'), 
                                ('PH2FOFCWT', 'PHWT'), 
                                ('FOFCWT', 'DELFWT'), 
                                ('PHFOFCWT', 'PHDELWT')])
        elif styles == 'scala_refmac':
            for label in self.GetLabels():
                self.RenameColumn((label, label.replace('_New','')))
            self.RenameColumns([('F', 'FP'),
                                ('SIGF', 'SIGFP'),
                                ('FreeR_flag', 'FREE')])

    def GetLabelListByType(self, coltype):
        colist = []
        for col in self.columns:
            if col.get_type() == coltype:
                colist.append(col.get_label())
        return colist

# --- All things test set

    def GetTestStyle(self, label):
        ''' Try to figure out the test set type.  Return one of the following 
            strings: "refmac", "cns", "cnsinv", "none" or "unknown".  Returns 
            None if the column is not integer type.'''
        if self.GetColumn(label).get_type() == 'I':
            flags = array(self.GetReflectionColumn(label))
            mif, maf = nanmin(flags), nanmax(flags)
            if maf-mif > 1:
                return 'refmac'
            elif maf-mif == 0:
                return 'none'
            elif mif == 0:
                n0, n1 = sum(flags==0), sum(flags==1)
                if n0 > n1:
                    return 'cns'
                else:
                    return 'cnsinv'
            else:
                return 'unknown'
        else:
            return None

    def GetTestIndex(self, label, fBoolean=False):
        ''' 
            Returns the index of reflections that belong to the test set.
            Attempts automatic determination of the test set type, 
            returns None if that fails.
            If fBoolean is True, returns a vector of boolean values 
            instead of integer index.
        '''
        style = self.GetTestStyle(label)
        flags = array(self.GetReflectionColumn(label))
        if style in ['refmac', 'cnsinv']:
            if fBoolean:
                return flags==0
            else:
                return nonzero(flags==0)[0]
        elif style == 'cns':
            if fBoolean:
                return flags==1
            else:
                return nonzero(flags==1)[0]

    def GetWorkIndex(self, label, fBoolean=False):
        ''' Returns the index of reflections that belong to the working set.
            Attempts automatic determination of the test set type, returns None
            if that fails. '''
        style = self.GetTestStyle(label)
        flags = array(self.GetReflectionColumn(label))
        if style in ['refmac', 'cnsinv']:
            if fBoolean:
                return flags!=0
            else:
                return nonzero(flags!=0)[0]
        elif style == 'cns':
            if fBoolean:
                return flags==0
            else:
                return nonzero(flags==0)[0]

    def GenerateFreeFlag(self, free_label, fraction = 0.05, style='refmac'):
        ''' Generates a test set flag column (the column itself must exist 
            before the method is called).  Both "refmac" and "cns" styles are
            permitted.  For refmac style, if fraction is >1 it is rounded to
            the nearest integer and used for the number of integer values
            spanning [0, ceil(fraction)].  To get a standard refmac style 
            5% test set, use fraction=20.  '''
        if free_label not in self.GetLabels():
            self.append_column(zeros(self.nref), free_label, 'I', 1)
        free_col = self.GetColumnIndex(free_label)
        if style == 'refmac':
            if fraction > 1:
                for (i, value) in enumerate(randint(0, ceil(fraction).astype(int), self.nref).tolist()):
                    self.reflections[i][free_col] = value
            else:
                for i in range(self.nref):
                    self.reflections[i][free_col] = 2*int(random.random()>fraction)
        elif style == 'cns':
            for i in range(self.nref):
                self.reflections[i][free_col] = int(random.random()<fraction)
        elif style == 'cnsinv':
            for i in range(self.nref):
                self.reflections[i][free_col] = 1 - int(random.random()<fraction)

    def GetTestFraction(self, label):
        ''' Determine the fraction of the test set as defined by the column with
            the specified label. Attempts automatic determination of the test 
            set type, returns None if that fails. '''
        style = self.GetTestStyle(label)
        flags = array(self.GetReflectionColumn(label))
        if style in ['refmac', 'cnsinv']:
            return float(sum(flags==0))/float(len(flags))
        elif style == 'cns':
            return float(sum(flags==1))/float(len(flags))

    def GetTestInfo(self, label):
        ''' Returns tuple that contains information about test set.
            (style, frac, numf) contains test set type, fraction of free
            reflections and their number. Returns None if the test
            set style cannot be determined.'''
        style = self.GetTestStyle(label)
        if style:
            flags = array(self.GetReflectionColumn(label))
            if style in ['refmac', 'cnsinv']:
                numf = sum(flags==0)
                frac = float(numf)/float(len(flags))
            elif style == 'cns':
                numf = sum(flags==1)
                frac = float(numf)/float(len(flags))
            return (style, frac, numf)

    def RegenerateFreeFlag(self, label):
        ''' Regenerates the test set flag column using the automatically
            determined type and fraction.  Not sure what it will return if
            automatic determination fails. '''
        self.GenerateFreeFlag(free_label=label, 
                              fraction=self.GetTestFraction(label), 
                              style=self.GetTestStyle(label))

    def RotateFreeFlag(self, label, shft=1):
        ''' Rotates the test test by adding shft to the value for each
            reflection, wrapped by the range of flags.  This can be
            used to make refinement software to use a different test
            set.  '''
        flags = array(self.GetReflectionColumn(label)).astype(int)
        trng = int(flags.ptp()+1)
        flags = ((flags+shft).astype(int))%trng
        self.SetReflectionColumn(label, flags)

    def ShrinkTestSet(self, label, new_fraction):
        ''' Reduces the size of the test set by resetting a random subset of the
            test set reflections. '''
        old_fraction = self.GetTestFraction(label)
        if old_fraction > new_fraction:
            test_index = self.GetTestIndex(label)
            Ntest = len(test_index)
            Nomit = int(Ntest * (old_fraction-new_fraction)/old_fraction)
            free_col = self.GetColumnIndex(label)
            style = self.GetTestStyle(label)
            if style == 'refmac':
                for i in test_index[permutation(Ntest)[:Nomit]]:
                    self.reflections[i][free_col] = 2
            elif style == 'cns':
                for i in test_index[permutation(Ntest)[:Nomit]]:
                    self.reflections[i][free_col] = 0
            elif style == 'cnsinv':
                for i in test_index[permutation(Ntest)[:Nomit]]:
                    self.reflections[i][free_col] = 1

    def GrowTestSet(self, label, new_fraction):
        ''' Increases the size of the test set by adding a random subset of the
            work set reflections.  Keeps the original test set reflections.'''
        old_fraction = self.GetTestFraction(label)
        if old_fraction < new_fraction:
            work_index = self.GetWorkIndex(label)
            Nwrk = len(work_index)
            Nadd = int(self.nref * (new_fraction-old_fraction))
            free_col = self.GetColumnIndex(label)
            style = self.GetTestStyle(label)
            if style in ['refmac', 'cnsinv']:
                for i in work_index[permutation(Nwrk)[:Nadd]]:
                    self.reflections[i][free_col] = 0
            elif style == 'cns':
                for i in work_index[permutation(Nwrk)[:Nadd]]:
                    self.reflections[i][free_col] = 1

    def TwinnedReflectionList(self, twinop, subset):
        mats = symoputils.symop_rots(self.GetSymInfo().get_spacegroup_number(),
                                                                        True)
        twinmat = symoputils.symop_rot(twinop)
        hkl = self.GetHKLArray()[subset]
        hklt = twinmat*matrix(hkl).T
        hshlst = []
        if self.hsh == None:
            self.hklash()
        for symt in mats:
            hklts = symt*hklt
            hsh_pls = self.GetHKLhash(hkl=array(hklts))
            hshlst += intersect1d(hsh_pls, self.hsh).tolist()
            hsh_mns = self.GetHKLhash(hkl=-array(hklts))
            hshlst += intersect1d(hsh_mns, self.hsh).tolist()
        return setdiff1d(self.hashindex(hshlst), subset)

    def TestSetAppend(self, label, listik):
        free_col = self.GetColumnIndex(label)
        style = self.GetTestStyle(label)
        if style in ['refmac',  'cnsinv']:
            for i in listik:
                self.reflections[i][free_col] = 0
        elif style == 'cns':
            for i in listik:
                self.reflections[i][free_col] = 1

    def TwinTestSet(self, label, twinop):
        test_index = self.GetTestIndex(label)
        twnlst = self.TwinnedReflectionList(twinop, test_index)
        Ntst = len(test_index)
        Ntwn = len(twnlst)
        self.ShrinkTestSet(label, self.GetTestFraction(label)*Ntst/(Ntst+Ntwn))
        test_index = self.GetTestIndex(label)
        twnlst = self.TwinnedReflectionList(twinop, test_index)
        self.TestSetAppend(label, twnlst)

# ---

    def ScaleColumn(self, label, factor=1.0):
        col = self.GetColumnIndex(label)
        for i in range(self.nref):
            self.reflections[i][col] *= factor

    def ShakeColumn(self, fp_label='FP', sigfp_label='SIGFP', RSF=2.0):
        fp_col = self.GetColumnIndex(fp_label)
        sigfp_col = self.GetColumnIndex(sigfp_label)
        for i in range(self.nref):
            fp_shk = random.gauss(self.reflections[i][fp_col], 
                                  RSF*self.reflections[i][sigfp_col])
            while fp_shk < 0:
                fp_shk = random.gauss(self.reflections[i][fp_col], 
                                      RSF*self.reflections[i][sigfp_col])
            self.reflections[i][fp_col] = fp_shk
        self.ResetMinMax(fp_label)

    def projected_sigma(self, params=(1.0, 1.0, 0.0, 0.0), 
                              labels=['FP', 'SIGFP']):
        ''' Return the projected sigmas using the set of parameters.  
            Specifically, params=(rsf0, Drsf, s_crit, gamma), and the projected 
            sigma s is calculated from original sigma s0 as follows
                s = s0 * (rsf0 + Drsf / (1 + (s_crit / s)^gamma)
            By default, one would get a fixed rsf=2.  The approximation is
            based on experiemental results and a good starting point is
                rsf0=1.0, s_crit~10.0, gamma=2.
            The Drsf varies, but a good estimate is 2-3. '''
        fp, sigfp = self.GetReflectionColumns(labels)
        (rsf0, Drsf, s_crit, gamma) = params
        rsf = rsf0 + Drsf / (1 + (s_crit * sigfp / fp)**gamma)
        rsf[isinf(rsf0)] = rsf0
        return rsf * sigfp

    def predict_drsf(self, rsf=2.0, rsf0=1.0, s_crit=0.0, gamma=0.0, 
                                        labels=['FP', 'SIGFP']):
        ''' Calculate the value of Drsf (see projected_sigma() method for 
            further description) that will produce the overall inflation
            factor rsf for the current dataset. '''
        fp, sigfp = self.GetReflectionColumns(labels)
        factor = sigfp/(1 + (s_crit * sigfp / fp)**gamma)
#        factor[isinf(factor)] = rsf0
        return (rsf-rsf0)*nanmean(sigfp)/nanmean(factor)

    def inflate_sigmas(self, rsf=2.0, rsf0=1.0, s_crit=10.0, gamma=2.0, 
                                       labels=['FP', 'SIGFP']): 
        ''' Inflate the sigmas using the standard approximation.  This uses the
            same equation as in the projected_sigma() method but individual 
            parameters are supplied and target rsf is use instead of Drsf (which
            is calculated using the predict_drsf() method. '''
        fp, sigfp = self.GetReflectionColumns(labels)
        Drsf = ((rsf-rsf0)*nanmean(sigfp) / 
                            nanmean(sigfp/(1 + (s_crit * sigfp / fp)**gamma)))
        rsf = rsf0 + Drsf / (1 + (s_crit * sigfp / fp)**gamma)
        rsf[isinf(rsf0)] = rsf0
        return rsf * sigfp

    def AddNoise(self, label='FP', sigma=1.0):
        self.SetReflectionColumn(label, 
                              sigma * array(self.GetReflectionColumn(label)) + 
                                        randn(self.GetReflectionNumber()))

    def wild_bootstrap(self, label='FP', fclabel='FC_ALL', sigma=1.0):
        ''' Does nothing at the moment. '''
        pass

    def __dset_id_verify_(self, dset_id):
        if dset_id not in list(self.dcell.keys()):
            return min(self.dcell.keys())
        else:
            return dset_id

    def shell_splitter(self, num=10, dset_id=-1):
        ''' Better way to split reflection into shells. 
            Returns the list of index arrays for each resolution shell. Notice
            that shells are in order from high to low resolution.'''
        d = array(self.GetResolutionColumn(dset_id))
        ind = ((num*d.argsort().argsort()).astype(float)/len(d)).astype(int)
        return [nonzero(ind==k)[0] for k in range(num)]

    def shell_shuffle(self, labels, num=10, d_high=0.0, d_low=float('inf'), dset_id=-1):
        ''' Shuffles reflection values within each resolution shell.
            Columns listed in labels will have the values shuffled within 
            every shell. '''
        splitter = self.shell_splitter(num, dset_id)
        vectors = self.GetReflectionColumns(labels)
        d = self.GetResolutionColumn()
        dmin, dmax = array([(d[k].min(), d[k].max()) for k in splitter]).T
        span = nonzero([(d[k].min()>=d_high and d[k].max()<=d_low) for k in splitter])[0]
        for ind in array(splitter)[span]:
            for vector in vectors:
                vector.put(ind, vector[permutation(ind)])
        for (i,label) in enumerate(labels):
            self.SetReflectionColumn(label, vectors[i])
        return (d[splitter[span[0]]].min(),  d[splitter[span[-1]]].max())

    def ResetMinMax(self, label):
        self.GetColumn(label).set_min_max(self.ColumnMinMax(label))

    def DeleteColumn(self, label):
        colind = self.GetColumnIndex(label)
        if colind == None:
            raise KeyError('No such column: |'+label+'|')
        self.columns.pop(colind)
        reflections = array(self.reflections).T.tolist()
        reflections.pop(colind)
        self.reflections = array(reflections).T.tolist()
        self.ncol -= 1

    def ColumnMinMax(self, label):
        col, x = self.GetColumnIndex(label), []
        for reflection in self.reflections:
            if not self.isnan(reflection[col]):
                x.append(reflection[col])
        return (min(x), max(x))

    def ColumnMissingNumber(self, label):
        ''' 
            Returns the number of reflections in the column marked as missing.
        '''
        col, x = self.GetColumnIndex(label), 0
        for reflection in self.reflections:
            if self.isnan(reflection[col]):
                x += 1
        return x

    def ColumnPresentNumber(self, label):
        '''
            Returns the number of resflections present (not missing) in the 
            column.
        '''
        col, x = self.GetColumnIndex(label), 0
        for reflection in self.reflections:
            if not self.isnan(reflection[col]):
                x += 1
        return x

    def ColumnMissIndex(self, label):
        col, x = self.GetColumnIndex(label), []
        for (i, reflection) in enumerate(self.reflections):
            if self.isnan(reflection[col]):
                x.append(i)
        return x

    def ColumnCompleteness(self, label):
        ''' Returns completeness of the label column as a fraction. '''
        return 1.0-float(self.ColumnMissingNumber(label))/float(self.nref)

    def HasLabel(self, label):
        ''' True if label column is present, false otherwise.'''
        return label in self.GetLabels()

    def BuildIntensities(self, fplabel='FP', sigfplabel='SIGFP', 
                                                ilabel='I', sigilabel='SIGI'):
        ''' Fills the intensity column by calculating the square of amplitudes.
            Fills the sigI column as 2*F*sigF.
            Overwrites the I/SIGI columns if they are already present.'''
        fp = array(self.GetReflectionColumn(fplabel))
        sigfp = array(self.GetReflectionColumn(sigfplabel))
        Icol = fp**2
        sigIcol = 2.0*sigfp*fp
        normfactor = nanmean(fp)/nanmean(Icol)
        Icol *= normfactor
        sigIcol *= normfactor
        if self.HasLabel(ilabel):
            self.SetReflectionColumn(ilabel, Icol)
        else:
            self.append_column(Icol, ilabel, 'J', 
                                       self.GetColumn(fplabel).get_dataset_id())
        if self.HasLabel(sigilabel):
            self.SetReflectionColumn(sigilabel, sigIcol)
        else:
            self.append_column(sigIcol, sigilabel, 'Q', 
                                    self.GetColumn(sigfplabel).get_dataset_id())

    def Truncate(self, ctruncate_executable, ilabel='I', sigilabel='SIGI', 
                                                   freelabel='FREE', nres=None):
        ''' Runs truncate to convert intensities to amplitudes.
            Environment must be configured to execute ctruncate.
            Notice that column labels will be converted to FP,SIGFP.
            IMPORTANT!!! This returns a new mtz object with truncated 
            intensities (or None if ctruncate fails).'''
        tmpfolder = tempfile.mkdtemp()
        hklin = os.path.join(tmpfolder,'hklin.mtz')
        self.write(hklin)
        hklout = os.path.join(tmpfolder,'hklout.mtz')
        flog = open(os.path.join(tmpfolder,'ctruncate.log'),'w')
        if nres:
            cmrun = subprocess.Popen([ctruncate_executable,
                                   '-hklin',hklin,
                                   '-hklout',hklout,
                                   '-nres',str(nres), 
                                   '-colin', '/*/*/['+ilabel+','+sigilabel+']'],
                stdin=open("/dev/null"), stdout=flog, stderr=subprocess.STDOUT)
        else:
            cmrun = subprocess.Popen([ctruncate_executable,
                    '-hklin',hklin,
                    '-hklout',hklout, 
                    '-colin', '/*/*/['+ilabel+','+sigilabel+']'], 
                stdin=open("/dev/null"), stdout=flog, stderr=subprocess.STDOUT)
        cmrun.wait()
        if cmrun.returncode:
            flog.close()
            return None
        mtz = read_mtz_file(hklout)
        mtz.RenameColumns([['F', 'FP'], ['SIGF', 'SIGFP']])
        mtz.append_column(self.GetReflectionColumn(freelabel), freelabel, 'I', 
                                    self.GetColumn(freelabel).get_dataset_id())
        shutil.rmtree(tmpfolder)
        return mtz

    def Shruncate(self, ilabel='I', sigilabel='SIGI', 
                                            fplabel='FP', sigfplabel='SIGFP'):
        ''' Converts intensities to amplitudes using internal algorithm.
            See Sivia, David, Acta Cryst A50:703 for details.'''
        Icol = array(self.GetReflectionColumn(ilabel))
        sigI = array(self.GetReflectionColumn(sigilabel))
        fp = 0.5*sqrt(2.0*Icol+sqrt(4.0*Icol*Icol+8.0*sigI*sigI))
        sigfp = 1.0/sqrt(1/fp**2 + 2.0*(3.0*fp**2-Icol)/sigI**2)
        if self.HasLabel(fplabel):
            self.SetReflectionColumn(fplabel, fp)
        else:
            self.append_column(fp, fplabel, 'F', 
                                        self.GetColumn(ilabel).get_dataset_id())
        if self.HasLabel(sigfplabel):
            self.SetReflectionColumn(sigfplabel, sigfp)
        else:
            self.append_column(sigfp, sigfplabel, 'Q', 
                                    self.GetColumn(sigilabel).get_dataset_id())

    def isnan(self, value):
        return str(value)=='nan'

    def omit(self, fraction, fp_label='FP', sigfp_label='SIGFP'):
        ''' Reset fraction of the reflections to NaN.  The initial completeness 
            is checked and this function does nothing if more than fraction of
            reflections is already missing.  Omission is permanent and cannot be
            undone, make a copy of the class if restoration of the original 
            condition is desired.  Returns the new completeness.'''
        extra_fraction = fraction + self.ColumnCompleteness(fp_label) - 1
        if extra_fraction > 0:
            nan = float('nan')
            fp_col = self.GetColumnIndex(fp_label)
            sigfp_col = self.GetColumnIndex(sigfp_label)
            for i in range(self.nref):
                if random.random() <= extra_fraction:
                    self.reflections[i][fp_col] = nan
                    self.reflections[i][sigfp_col] = nan
        self.ResetMinMax(fp_label)
        self.ResetMinMax(sigfp_label)
        return self.ColumnCompleteness(fp_label)

    def Qml_phenix(self, labels=('FOBS', 'FMODEL', 'ALPHA', 'BETA', 'EPSILON')):
        fobs, fcalc, alpha, beta, eps = self.GetReflectionColumns(labels)
        esbs = eps*beta
        sigm = sqrt(esbs)
        p = fobs/sigm
        fstar = sigm*mu_acentric(p)/alpha
        ind_centric = nonzero(eps>1)[0]
        fstar[ind_centric] = (sigm[ind_centric] * 
                                mu_centric(p[ind_centric]) / 
                                  alpha[ind_centric])
        wstar = alpha**2*nu_acentric(p)/esbs
        wstar[ind_centric] = (0.5*alpha[ind_centric]**2 * 
                                nu_centric(p[ind_centric]) / 
                                  esbs[ind_centric])
        return wstar*(fcalc-fstar)**2

    def Qml_clipper(self, 
                    labels=('FOBS','SIGFOBS','FMODEL','WM','SIGM','EPSILON')):
        fobs, sigf, fcalc, wm, sigm, eps = self.GetReflectionColumns(labels)
        epsc = eps
        ind_centric = nonzero(eps>1)[0]
        epsc[ind_centric] = 2.0*eps
        sigma = 2.0*sigf**2+eps*sigm**2
        qml = (log(sigma) + 
                (fobs**2+(wm*fcalc)**2)/sigma - 
                  log(i0(2.0*fobs*wm*fcalc/sigma)))
        qml[ind_centric] = (0.5*log(sigma) + 
                             (fobs**2+(wm*fcalc)**2)/sigma - 
                               log(cosh(2.0*fobs*wm*fcalc/sigma)))
        return qml

    def write(self, fname):
        fout = open(fname,'wb')
        mtzstamp = binarray.array('B')
        mtzstamp.fromstring('MTZ ')
        mtzstamp.tofile(fout)
        headerposition = binarray.array('I')
        headerposition.append(self.nref*self.ncol+21)
        headerposition.tofile(fout)
        machinestamp =  binarray.array('B')
        machinestamp.extend([68, 65]+[0]*70)
        machinestamp.tofile(fout)
        refs = binarray.array('f')
        for reflection in self.reflections:
            refs.extend(reflection)
        refs.tofile(fout)
        fout.write(('VERS '+self.mtzversion).ljust(80).encode())
        fout.write(('TITLE '+self.mtztitle.ljust(70)[:70]).ljust(80).encode())
        fout.write(('NCOL %8d %12d %8d' 
                                % (self.ncol, self.nref, self.nbat)).ljust(80).encode())
        fout.write(('CELL  %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f' 
                                % (self.globalcell.get_cell())).ljust(80).encode())
        fout.write(('SORT  %3d %3d %3d %3d %3d' 
                                % tuple(self.sortorder)).ljust(80).encode())
        fout.write(("SYMINF  %2d %2d %c %5d           '%s' %s" 
                                % (self.syminfo.get_nsym(), 
                                    self.syminfo.get_nprim(), 
                                     self.syminfo.get_lattice_type(), 
                                      self.syminfo.get_spacegroup_number(), 
                                       self.syminfo.get_spacegroup_name(), 
                                self.syminfo.get_pointgroup_name())).ljust(80).encode())
        for symop in self.symm:
            fout.write(('SYMM '+symop.get_line()).ljust(80).encode())
        fout.write(('RESO %9.6f    %9.6f' 
                                % (self.lowres, self.highres)).ljust(80).encode())
        fout.write(('VALM %s' % self.missflag).ljust(80).encode())
        for col in self.columns:
            fout.write(('COLUMN ' + col.get_label().ljust(30) + 
                                                ' %c  %16.4f  %16.4f %4d' 
                                % (col.get_type(), 
                                    col.get_min(), 
                                     col.get_max(),
                                      col.get_dataset_id())).ljust(80)[:80].encode())
        fout.write(('NDIF    %5d' % self.ndat).ljust(80).encode())
        for dset_id in self.projects.keys():
            fout.write(('PROJECT %7d %s' 
                            % (dset_id, self.projects[dset_id])).ljust(80)[:80].encode())
            fout.write(('CRYSTAL %7d %s' 
                            % (dset_id, self.crystals[dset_id])).ljust(80)[:80].encode())
            fout.write(('DATASET %7d %s' 
                            % (dset_id, self.datasets[dset_id])).ljust(80)[:80].encode())
            fout.write(('DCELL   %7d  %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f' 
                            % (tuple([dset_id]) + 
                                    self.dcell[dset_id].get_cell())).ljust(80).encode())
            fout.write(('DWAVEL  %7d %10.5f' 
                            % (dset_id, self.dwavel[dset_id])).ljust(80).encode())
        fout.write('END'.ljust(80).encode())
        for history_item in self.history:
            fout.write(history_item)
        fout.write('MTZENDOFHEADERS'.ljust(80).encode())
        fout.close()

    def CNSWrite(self, fname, fp='FP', sigfp='SIGFP', 
                                            free='FREE', TestConvertFlag=False):
        fout = open(fname,'wb')
        fobs = self.GetReflectionColumn(fp)
        sigma = self.GetReflectionColumn(sigfp)
        if TestConvertFlag:
            test = []
            for value in self.GetReflectionColumn(free):
                test.append(int(not value))
        else:
            test = self.GetReflectionColumn(free)
        h = self.GetReflectionColumn('H')
        k = self.GetReflectionColumn('K')
        l = self.GetReflectionColumn('L')
        hklfst = []
        for i in range(len(h)):
            if (fobs[i] == fobs[i] and 
                    sigma[i] == sigma[i] and 
                        test[i] == test[i]):
                hklfst.append((h[i], k[i], l[i], fobs[i], sigma[i], test[i]))
        fout.write('NREFlection=%10d\n' % len(hklfst))
        fout.write('ANOMalous=FALSe { equiv. to HERMitian=TRUE}\n')
        fout.write('DECLare NAME=FOBS                   DOMAin=RECIprocal   TYPE=REAL END\n')
        fout.write('DECLare NAME=SIGMA                  DOMAin=RECIprocal   TYPE=REAL END\n')
        fout.write('DECLare NAME=TEST                   DOMAin=RECIprocal   TYPE=INTE END\n')
        for item in hklfst:
            fout.write('INDE %5d%5d%5d FOBS=%10.3f SIGMA=%10.3f TEST=%10d\n' 
                                                                        % item)
        fout.close()

    def copy(self):
        return copy.deepcopy(self)


    def printout(self, what):
        ''' Returns the strings containing various informative messages about
            the data.  Following keywords are accepted (with CCP4-like
            convention, i.e. that case doesn't matter and the first four letters
            defeine the result):
                RESLimits       - resolution limits in angstroms
                LABEls          - column labels
                TYPEs           - column types
                CELL            - unit cell parameters
                SPGRoup         - space group
        '''
        key = what.upper()[:4]
        if key == 'RESL':
            line = 'Resolution limits %6.1f - %6.2f\n' % (
                                    pow(self.lowres, -0.5),
                                    pow(self.highres, -0.5) )
        elif key == 'LABE':
            line = 'Columns:\n' + ' '.join(self.GetLabels()) + '\n'
        elif key == 'TYPE':
            line = 'Types:\n' + ' '.join(self.GetTypes()) + '\n'
        elif key == 'CELL':
            line = 'Unit cell paramaters:\n' + '%6.3f '*6 % self.GetGlobalCell().get_cell() + '\n'
        elif key == 'SPGR':
            line = self.GetSymInfo().get_spacegroup_name() + '\n'
        else:
            raise KeyError(what+' is not a recognizable pymtz.printout key.')
        return line

class MTZdsets:
    ''' 
        Dataset information class, provides access to project, crystal, dataset,
        cell and wavelength info.
    '''
    def __init__(self):
        self.dsets = {'project' : {}, 
                      'crystal' : {}, 
                      'dataset' : {},
                      'dcell'   : {}, 
                      'dwavel'  : {}}

    def insert(self, dset, key, value):
        self.dsets[key][dset] = value

class MTZheader:
    ''' This class may be useful when parsing mtz file for various parameters
        stored in the header without the need to read/store the actual 
        reflection data. 
        Header data are stored in mtzHeader.records dictionary that have the 
        following mostly self-explanatory keys that match the record labels
        in mtz file:
        vers
        title
        ncol        [N_columns, N_reflections, N_batches]
        cell        global cell [a,b,c,al,be,ga]
        sort        list defining the sort order
        syminf      SymmInfo class
        symm        List of SymmetryOperator instances
        reso        [low_d*, high_d*]
        valm
        ndif
        
        '''

    def __init__(self, fname):
        fin = open(fname,'rb')
        mtzstamp = readarray(fin, 'B', 4)
        if mtzstamp.tostring() != 'MTZ ':
            raise TypeError(fname+' is not an MTZ-file')
        headerposition = readarray(fin)
        machinestamp =  readarray(fin, 'B', 4)
        self.records = {
                        'symm' : [], 
                        'col' : [],
                        'dsets' : MTZdsets()
                       }
        fin.seek((headerposition[0]-1)*4)
        record_flag = True
        while record_flag:
            record = readarray(fin, 'B', 80)
            record_flag = self.assign_record(record.tostring())
            if record_flag == 'END':
                historyposition = fin.tell()
                record_flag = True
        nhis = int((fin.tell()-historyposition)/80)-1
        f_batches = False
        fin.seek(historyposition)
        for i in range(nhis):
            record = readarray(fin, 'B', 80)
            hisline = record.tostring()
            if hisline[:7] == 'MTZBATS':
                f_batches = True
                break
            else:
                self.history.append(record.tostring())
        if f_batches:
            while True:
                record = readarray(fin, 'B', 80)
                keys = record.tostring().split()
                if keys[0] == 'BH':
                    curb = int(keys[1])
                    self.batchlist[curb] = [keys[2:]]
                elif keys[0] == 'TITLE':
                    self.batchlist[curb].append(' '.join(keys[1:]))
                    self.batchlist[curb].append(MTZoriblock(fin))
                    break
        fin.close()

    def assign_record(self, line):
        ''' Assigns an MTZ record based on the content of the string. '''
        key = line.split()[0]
        value = line.split(key)[1]
        if key == 'VERS':
            self.records[key.lower()] = value.strip().ljust(10)
        elif key == 'TITLE':
            self.records[key.lower()] = value.strip().ljust(70)
        elif key in ['NCOL', 'SORT']:
            self.records[key.lower()] = [int(x) for x in value.split()]
        elif key == 'CELL':
            self.records[key.lower()] = [float(x) for x in value.split()]
        elif key == 'SYMINF':
            self.records[key.lower()] = SymmInfo(value)
        elif key == 'SYMM':
            self.records[key.lower()].append(SymmetryOperator([x.strip() for x in value.split(',')]))
        elif key == 'RESO':
            self.records[key.lower()] = sorted(map(float, value.split()[:2]))
        elif key == 'VALM':
            self.records[key.lower()] = value.strip()
        elif key == 'COLUMN' or key == 'COL':
            self.records['col'].append(MTZcolumn(value.split()))
        elif key == 'NDIF':
            self.records[key.lower()] = int(value.split()[0])
        elif key in ['PROJECT', 'CRYSTAL', 'DATASET']:
            dset = value.split()[0]
            name = value.split(dset)[1].strip()
            self.records['dsets'].insert(dset, key.lower(), name)
        elif key == 'DCELL':
            dset = value.split()[0]
            cellprms = [float(x) for x in value.split()[1:]]
            self.records['dsets'].insert(dset, key.lower(), cellprms)
        elif key == 'DWAVEL':
            dset = value.split()[0]
            wavelength = float(value.split()[1])
            self.records['dsets'].insert(dset, key.lower(), wavelength)
        elif key == 'BATCH':
            self.AddBatches(line[len(key):].split())
        elif key == 'END':
            return key
        elif key == 'MTZENDOFHEADERS':
            return False
        return True


