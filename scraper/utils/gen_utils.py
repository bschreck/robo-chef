import cPickle as pickle
import shutil
import os

def updatePickleFile(object2save,pf):
    #copy old pickle file contents to temporary file
    #save object to pickle file,
    #delete the temporary file
    temp_pf = 'pickle_files/temp.p'
    if not os.path.isfile(pf):
        f = open(pf, 'w+')
        f.close()
    if not os.path.isfile(temp_pf):
        f = open(temp_pf, 'w+')
        f.close()

    shutil.copyfile(pf, temp_pf)
    try:
        pickle.dump(object2save, open(pf,'wb'))
    except:
        shutil.copyfile(temp_pf, pf)
    os.remove(temp_pf)


def loadObjectFromPickleFile(pf):
    if os.path.isfile(pf):
        try:
            p_object = pickle.load(open(pf, 'rb'))
        except:
            return None
        if p_object:
            return p_object
    return None

def replaceNonAscii(string):
    return ''.join([i if ord(i) < 128 else ' ' for i in string])
