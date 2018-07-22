''' The module provides several methods to manipulate symmetry operators.
    For the data that defines the symmetries see SpaceGroups module. 
    The Spacegroup.symops dictionary is imported by this module and
    contains inline methods that define symmetry operators.  Same is
    assumed throughout the module - the symmetry operators are passed
    to individual methhods as inline methods, e.g.
    
    lambda x,y,z: (x,y,z)
    
    defines the identity operator.'''
from scipy import array, matrix

from SpaceGroups import symops

M2PI = - 360.0

def symop_tr(symm):
    '''Return the translation vector for the symmetry operator.'''
    return array(symm(0, 0, 0))

def symop_trs(symn, fidentity=False):
    ''' Returns the array of translation vectors for all the symmetry operators
        of the space group defined by number, symn. The fidentity flag defines
        if the identity operator is included in the list.'''
    trs = [array(symm(0, 0, 0)) for symm in symops[symn]]
    if not fidentity:
        trs.pop(0)
    return array(trs)

def symop_rot(symm):
    '''Return the rotation matrix for the symmetry operator.'''
    trvec = array(symm(0, 0, 0))
    return matrix([ array(symm(1, 0, 0))-trvec,
                    array(symm(0, 1, 0))-trvec,
                    array(symm(0, 0, 1))-trvec])

def symop_rots(symn, fidentity=False):
    ''' Returns the list of rotation matrices for all the symmetry operators
        of the space group defined by number, symn. The fidentity flag defines
        if the identity operator is included in the list.'''
    rots = [symop_rot(symm) for symm in symops[symn]]
    if not fidentity:
        rots.pop(0)
    return rots

def phase_shift(symm):
    '''Returns the phase shift in degrees for the symmetry operator.'''
    return M2PI*symop_tr(symm)

def phase_shifts(symn, fidentity=False):
    ''' Returns the list of phase shifts (in degrees) for all the symmetry 
        operators of the space group defined by number, symn. The fidentity 
        flag defines if the identity operator is included in the list.'''
    shfts = [M2PI*array(symm(0, 0, 0)) for symm in symops[symn]]
    if not fidentity:
        shfts.pop(0)
    return shfts
    
def symop_uniq_rots(symn, fidentity=False):
    ''' Returns the list of unique rotation matrices for the symmetry operators
        of the space group defined by number, symn. The fidentity flag defines
        if the identity operator is included in the list.'''
    rots = [symop_rot(symops[symn][0])]
    for symm in symops[symn][1:]:
        rot = symop_rot(symm)
        if abs(array(rots)-array(rot)).sum(2).sum(1).all():
            rots.append(rot)
    if not fidentity:
        rots.pop(0)
    return rots

def symop_uniq_rotrans(symn, fidentity=False):
    ''' Returns rotation/translation for the unique rotations matrices for the 
        symmetry operators of the space group defined by number, symn.  The
        method returns a tuple of lists - the rotation matrices and translation
        vectors. The fidentity flag defines if the identity operator is 
        included in the list.'''
    rots = [symop_rot(symops[symn][0])]
    trs = [symop_tr(symops[symn][0])]
    for symm in symops[symn][1:]:
        rot = symop_rot(symm)
        if abs(array(rots)-array(rot)).sum(2).sum(1).all():
            rots.append(rot)
            trs.append(array(symm(0, 0, 0)))
    if not fidentity:
        rots.pop(0)
        trs.pop(0)
    return rots, trs

def symop_uniq_rotshift(symn, fidentity=False):
    ''' Returns rotation/phaase shift for the unique rotations matrices for the 
        symmetry operators of the space group defined by number, symn.  The
        method returns a tuple of lists - the rotation matrices and phase shifts
        (in degrees). The fidentity flag defines if the identity operator is 
        included in the list.'''
    rots = [symop_rot(symops[symn][0])]
    shfts = [phase_shift(symops[symn][0])]
    for symm in symops[symn][1:]:
        rot = symop_rot(symm)
        if abs(array(rots)-array(rot)).sum(2).sum(1).all():
            rots.append(rot)
            shfts.append(phase_shift(symm))
    if not fidentity:
        rots.pop(0)
        shfts.pop(0)
    return rots, shfts


