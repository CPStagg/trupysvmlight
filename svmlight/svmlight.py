from ctypes import *
from collections import namedtuple
import pickle

# change this as necessary to the correct path to your svmlight.so lib
import distutils.sysconfig
sharedlibpath = distutils.sysconfig.get_python_lib() + "/svmlight.so"
print sharedlibpath
svm = CDLL(sharedlibpath)

VERSION = "V6.02"
VERSION_DATE = "14.08.08"

CFLOAT = c_float
FNUM   = c_long
FVAL   = c_float

MAXFEATNUM = 99999999

LINEAR  = 0
POLY    = 1
RBF     = 2
SIGMOID = 3

CLASSIFICATION = 1
REGRESSION     = 2
RANKING        = 3
OPTIMIZATION   = 4

method_dict = { CLASSIFICATION : 'classification', 
                REGRESSION : 'regression', 
                RANKING : 'ranking', 
                OPTIMIZATION : 'optimization' }

enum_dict = {}
for k, v in method_dict.items():
    enum_dict[v] = k

MAXSHRINK = 50000
# ----------------------------------------------

class WORD(Structure):
    _fields_ = [("wnum",   FNUM),
                ("weight", FVAL)]
# ----------------------------------------------

class SVECTOR(Structure):
    pass

SVECTOR._fields_ = [("words",       POINTER(WORD)),
                    ("twonorm_sq",  c_double),
                    ("userdefined", POINTER(c_char)),
                    ("kernel_id",   c_long),
                    ("next",        POINTER(SVECTOR)),
                    ("factor",      c_double)]
# ----------------------------------------------

class DOC(Structure):
    _fields_ = [("docnum",     c_long),
                ("queryid",    c_long),
                ("costfactor", c_double),
                ("slackid",    c_long),
                ("fvec",       POINTER(SVECTOR))]
# ----------------------------------------------

class LEARN_PARM(Structure):
    _fields_ = [("type",                  c_long),
                ("svm_c",                 c_double),
                ("eps",                   c_double),
                ("svm_costratio",         c_double),
                ("transduction_posratio", c_double),
                ("biased_hyperplane",     c_long),
                ("sharedslack",           c_long),
                ("svm_maxqpsize",         c_long),
                ("svm_newvarsinqp",       c_long),
                ("kernel_cache_size",     c_long),
                ("epsilon_crit",          c_double),
                ("epsilon_shrink",        c_double),
                ("svm_iter_to_shrink",    c_long),
                ("maxiter",               c_long),
                ("remove_inconsistent",   c_long),
                ("skip_final_opt_check",  c_long),
                ("compute_loo",           c_long),
                ("rho",                   c_double),
                ("xa_depth",              c_long),
                ("predfile",              (c_char * 200)),
                ("alphafile",             (c_char * 200)),
                ("epsilon_const",         c_double),
                ("epsilon_a",             c_double),
                ("opt_precision",         c_double),
                ("svm_c_steps",           c_long),
                ("svm_c_factor",          c_double),
                ("svm_costratio_unlab",   c_double),
                ("svm_unlabbound",        c_double),
                ("svm_cost",              POINTER(c_double)),
                ("totwords",              c_long)]
# ----------------------------------------------

class KERNEL_PARM(Structure):
    _fields_ = [("kernel_type",     c_long),
                ("poly_degree",     c_long),
                ("rbf_gamma",       c_double),
                ("coef_lin",        c_double),
                ("coef_const",      c_double),
                ("custom",          (c_char * 50))]
# ----------------------------------------------

class MODEL(Structure):
    _fields_ = [("sv_num",          c_long),
                ("at_upper_bound",  c_long),
                ("b",               c_double),
                ("supvec",          POINTER(POINTER(DOC))),
                ("alpha",           POINTER(c_double)),
                ("index",           POINTER(c_long)),
                ("totwords",        c_long),
                ("totdoc",          c_long),
                ("kernel_parm",     KERNEL_PARM),
                ("loo_error",       c_double),
                ("loo_recall",      c_double),
                ("loo_precision",   c_double),
                ("xa_error",        c_double),
                ("xa_recall",       c_double),
                ("xa_precision",    c_double),
                ("lin_weights",     POINTER(c_double)),
                ("maxdiff",         c_double)]
# ----------------------------------------------

class QP(Structure):
    _fields_ = [("opt_n",     c_long),
                ("opt_m",     c_long),
                ("opt_ce",    POINTER(c_double)),
                ("opt_ce0",   POINTER(c_double)),
                ("opt_g",     POINTER(c_double)),
                ("opt_g0",    POINTER(c_double)),
                ("opt_xinit", POINTER(c_double)),
                ("opt_low",   POINTER(c_double)),
                ("opt_up",    POINTER(c_double))]
# ----------------------------------------------

class KERNEL_CACHE(Structure):
  _fields_ = [("index",         POINTER(c_long)),
              ("buffer",        POINTER(CFLOAT)),
              ("invindex",      POINTER(c_long)),
              ("active2totdoc", POINTER(c_long)),
              ("totdoc2active", POINTER(c_long)),
              ("lru",           POINTER(c_long)),
              ("occu",          POINTER(c_long)),
              ("elems",         c_long),
              ("max_elems",     c_long),
              ("time",          c_long),
              ("activenum",     c_long),
              ("buffsize",      c_long)]
# ----------------------------------------------

class TIMING(Structure):
    _fields_ = [("time_kernel",     c_long),
                ("time_opti",       c_long),
                ("time_shrink",     c_long),
                ("time_update",     c_long),
                ("time_model",      c_long),
                ("time_check",      c_long),
                ("time_select",     c_long)]
# ----------------------------------------------

class SHRINK_STATE(Structure):
    _fields_ = [("active",          POINTER(c_long)),
                ("inactive_since",  POINTER(c_long)),
                ("deactnum",        c_long),
                ("a_history",       POINTER(POINTER(c_double))),
                ("maxhistory",      c_long),
                ("last_a",          POINTER(c_double)),
                ("last_lin",        POINTER(c_double))]
# ----------------------------------------------

# specify return types for key methods in the svm C library
svm.sprod_ss.restype = c_double
svm.classify_example_linear.restype = c_double
svm.read_model.restype = POINTER( MODEL )
# ----------------------------------------------

''' This auxiliary function to svm_learn reads some parameters from the keywords to
 * the function and fills the rest in with defaults (from read_input_parameters()
 * in svm_learn_main.c:109).
 
 returns an int
'''

def read_learning_parameters( **kwds):
    
    client_data = CLIENTDATA()
    verbosity = client_data.pverb
    learn_parm = client_data.plearn
    kernel_parm = client_data.kparm

    learn_parm.predfile = "trans_predictions"
    learn_parm.alphafile = ""
    verbosity = 0
    learn_parm.biased_hyperplane = 1
    learn_parm.sharedslack = 0
    learn_parm.remove_inconsistent = 0
    learn_parm.skip_final_opt_check = 0
    learn_parm.svm_maxqpsize = 10
    learn_parm.svm_newvarsinqp = 0
    learn_parm.svm_iter_to_shrink = -9999
    learn_parm.maxiter = 100000
    learn_parm.kernel_cache_size = 40
    learn_parm.svm_c = 0.0
    learn_parm.eps = 0.1
    learn_parm.transduction_posratio = -1.0
    learn_parm.svm_costratio = 1.0
    learn_parm.svm_costratio_unlab = 1.0
    learn_parm.svm_unlabbound = 1E-5
    learn_parm.epsilon_crit = 0.001
    learn_parm.epsilon_a = 1E-15
    learn_parm.compute_loo = 0
    learn_parm.rho = 1.0
    learn_parm.xa_depth = 0
    kernel_parm.kernel_type = 0
    kernel_parm.poly_degree = 3
    kernel_parm.rbf_gamma = 1.0
    kernel_parm.coef_lin = 1
    kernel_parm.coef_const = 1
    kernel_parm.custom = "empty"
    learn_parm.type = CLASSIFICATION

    if "type" in kwds:
        typ = kwds["type"]
        if not typ in method_dict.values():
            raise Exception, "unknown learning type specified. Valid types are: 'classification', 'regression', 'ranking' and 'optimization'."

        learn_parm.type = enum_dict[ typ ]
            
    print 'Type:'
    print learn_parm.type

    if "kernel" in kwds:
        kernel = kwds["kernel"]
        if kernel == "linear":
            kernel_parm.kernel_type = LINEAR
        elif kernel == "polynomial":
            kernel_parm.kernel_type = POLY
        elif kernel == "rbf":
            kernel_parm.kernel_type = RBF
        elif kernel == "sigmoid":
            kernel_parm.kernel_type = SIGMOID
        else:
            raise Exception("unknown kernel type specified. Valid types are: 'linear', 'polynomial', 'rbf' and 'sigmoid'.")

    if "verbosity" in kwds:
        verbosity = kwds["verbosity"]

    if "C" in kwds:
        learn_parm.svm_c = kwds["C"]

    if "poly_degree" in kwds:
        kernel_parm.poly_degree = kwds["poly_degree"]

    if "rbf_gamma" in kwds:
        kernel_parm.rbf_gamma = kwds["rbf_gamma"]
   
    if "coef_lin" in kwds:
        kernel_parm.coef_lin = kwds["coef_lin"]

    if "coef_const" in kwds:
        kernel_parm.coef_const = kwds["coef_const"]
    
    if learn_parm.svm_iter_to_shrink == -9999:
        if kernel_parm.kernel_type == LINEAR:
            learn_parm.svm_iter_to_shrink = 2
        else:
            learn_parm.svm_iter_to_shrink = 100

    return client_data
# ----------------------------------------------

def count_doclist( doclist ):
    max_docs = len( doclist )
    max_words = 0
    for doctuple in iter( doclist ):
        words_list = doctuple[1]
        list_length = len( words_list )
        if list_length > max_words:
            max_words = list_length

    return ( max_docs, max_words )
# ----------------------------------------------

UnpackData = namedtuple('Unpackdata', 'words doc_label queryid slackid costfactor')
# ----------------------------------------------

WordTuple = namedtuple('WordTuple', 'wnum weight' )
# ----------------------------------------------

def unpack_document(docobj, max_words_doc):

    # We initialize these parameters with their default values, since we won't
    # be reading them from the feature pairs (don't really care).
    queryid, slackid, costfactor = 0, 0, 1

    if not isinstance( docobj, tuple ):
        raise Exception("document should be a tuple")

    label, words_list = docobj[0], docobj[1]
    if len( docobj ) > 2:
        queryid = docobj[2]

    if type(words_list) != list:
        raise Exception("expected list of feature pairs")

    words = [WordTuple( int( feat0 ), feat1 ) for
             feat0, feat1 in words_list[:max_words_doc]]

    # sentinel entry required by C code
    words.append( WordTuple( 0, 0.0 ) )

    returnval = UnpackData( words, label, queryid, slackid, costfactor )
    return returnval
# ----------------------------------------------

def create_example(docnum, queryid, slackid, costfactor, fvec):
    result = DOC()
    
    result.docnum = docnum
    result.queryid = queryid
    result.slackid = slackid
    result.costfactor = costfactor
    result.fvec = pointer( fvec )

    return result
# ----------------------------------------------

def create_example_from_unpack(unpackdata, currentsize, fvec):
    return create_example( currentsize, unpackdata.queryid, unpackdata.slackid, unpackdata.costfactor, fvec )
# ----------------------------------------------

class DOCLISTDATA(Structure):
    _fields_ = [("docs", POINTER(POINTER(DOC))),
                ("labels", POINTER(c_double)),
                ("totwords",  c_int),
                ("totdoc",  c_int)]
# ----------------------------------------------


def unpack_doclist( doclist ):
    try:
        doc_iterator = iter(doclist)
    except TypeError, te:
        raise Exception, "Not iterable"

    max_docs, max_words = count_doclist( doclist )

    tempdoclist = []
    templabellist = []
    totwords = 0
    for item in doc_iterator:

        unpackdata = unpack_document( item, max_words )
        if unpackdata.words:
            assert len(unpackdata.words) > 1
            candidatewords = unpackdata.words[-2].wnum
            totwords = max(totwords, candidatewords)

        docnum = unpackdata.doc_label

        fvec = create_svector( unpackdata.words, "", 1.0 )
  
        currentsize = len( tempdoclist )
        newdoc = create_example_from_unpack( unpackdata, currentsize, fvec )
        pdoc = pointer( newdoc )
        tempdoclist.append( pdoc )

        locdoc = tempdoclist[-1].contents

        templabellist.append( docnum )

    totdoc = len( doclist )

    carraydoc = ( POINTER( DOC ) * totdoc )()
    carraylabel = ( c_double * totdoc )() 
          
    for i, item in enumerate( tempdoclist ):
        carraydoc[ i ] = item
        
    for i, item in enumerate( templabellist ):
        carraylabel[ i ] = item

    result = DOCLISTDATA()

    result.docs = carraydoc
    result.labels = carraylabel
    result.totwords = totwords
    result.totdoc = totdoc
     
    return result
# ----------------------------------------------

def generate_C_string_from_python( pythonstring ):
    cstring = ( c_char * len( pythonstring ) )()
    cstring[:] = pythonstring
    return cstring
# ----------------------------------------------

def create_fixed_size_words( words ):
    result = (WORD * len( words ))()
    index = 0
    for item in words:
        result[ index ].wnum = item.wnum
        result[ index ].weight = item.weight
        index += 1

    return result
# ----------------------------------------------         

def create_svector( words, userdefined, factor ):

    result = SVECTOR()
    cwords = create_fixed_size_words( words )

    result.words = cwords
    result.twonorm_sq = svm.sprod_ss( pointer(result), pointer(result) )

    result.userdefined = generate_C_string_from_python( userdefined )
    result.factor = factor
    result.kernel_id = 0
    result.next = None
    
    return result
# ----------------------------------------------

class CLIENTDATA(Structure):
    _fields_ = [("pverb",  c_long),
            ("plearn", LEARN_PARM),
            ("kparm",  KERNEL_PARM)]
# ----------------------------------------------

SVMCallTuple = namedtuple('SVMCallTuple', 'doclistdata client_data kernel_cache model' )
# ----------------------------------------------

LearnResultsTuple = namedtuple( 'LearnResultsTuple', 'model docs totdoc' )
# ----------------------------------------------

def svm_learn( doclist, **kwds):
    
    client_data = read_learning_parameters( **kwds )

    doclistdata = unpack_doclist( doclist )

    model = MODEL()

    # this is a bit of a hack because of some slight nastiness in the C code, comparing
    # against the address of a null pointer
    kernel_cache = c_int( 0 )
    
    if client_data.kparm.kernel_type != LINEAR:
        kernel_cache = svm.kernel_cache_init( doclistdata.totdoc, 
                                              client_data.plearn.kernel_cache_size )

    svm_call_tuple = SVMCallTuple( doclistdata, client_data, kernel_cache, model )

    call_pattern = call_svm_method_with_null
    if client_data.plearn.type in [ REGRESSION, RANKING ]:
        call_pattern = call_svm_method_without_null

    svm_method_name = "svm_learn_" + method_dict[ client_data.plearn.type ]
    call_pattern( svm_method_name, svm_call_tuple )

    result = LearnResultsTuple( model, doclistdata.docs, doclistdata.totdoc )
    return result 
# ----------------------------------------------

def call_svm_method_with_null( method_name, svm_call_tuple ):

    method = getattr(svm, method_name)
    method( svm_call_tuple.doclistdata.docs, 
            svm_call_tuple.doclistdata.labels,
            svm_call_tuple.doclistdata.totdoc,
            svm_call_tuple.doclistdata.totwords,
            pointer( svm_call_tuple.client_data.plearn ),
            pointer( svm_call_tuple.client_data.kparm ),
            svm_call_tuple.kernel_cache,
            pointer( svm_call_tuple.model ),
            None )
# ----------------------------------------------

def call_svm_method_without_null( method_name, svm_call_tuple ):

    method = getattr(svm, method_name)
    method( svm_call_tuple.doclistdata.docs, 
            svm_call_tuple.doclistdata.labels,
            svm_call_tuple.doclistdata.totdoc,
            svm_call_tuple.doclistdata.totwords,
            pointer( svm_call_tuple.client_data.plearn ),
            pointer( svm_call_tuple.client_data.kparm ),
            pointer( svm_call_tuple.kernel_cache ),
            pointer( svm_call_tuple.model ) )
# ----------------------------------------------

def write_model( model, filename ):
    filename_as_c_string = generate_C_string_from_python( filename )
    svm.write_model( filename, pointer( model ) )
# ----------------------------------------------

def read_model( filename ):
    filename_as_c_string = generate_C_string_from_python( filename )
    pmodel = svm.read_model( filename )
    return pmodel.contents
# ----------------------------------------------

def svm_classify( model, doclist ):

    try:
        doc_iterator = iter(doclist)
    except TypeError, te:
        raise Exception, "Not iterable"

    docnum = 0
    dist = None

    has_linear_kernel = ( model.kernel_parm.kernel_type == 0 )

    if has_linear_kernel:
        svm.add_weight_vector_to_linear_model( pointer( model ) )

    max_docs, max_words = count_doclist( doclist )

    result = []
    for item in doc_iterator:
        unpackdata = unpack_document( item, max_words )

        if has_linear_kernel:

            for doc_item in unpackdata.words:
  
                if doc_item.wnum == 0:
                    #sentinel entry
                    break
                if doc_item.wnum > model.totwords:
                    doc_item.wnum = 0

            svector = create_svector( unpackdata.words, "", 1.0 )
            doc = create_example( -1, 0, 0, 0.0, svector )
            dist = svm.classify_example_linear( pointer(model), pointer(doc) )
        else:
            svector = create_svector( unpackdata.words, "", 1.0 )
            doc = create_example( -1, 0, 0, 0.0, svector )
            dist = svm.classify_example( pointer(model), pointer(doc) )

        result.append( dist )

    return result
# ----------------------------------------------
        

# -------------------- MAIN --------------------

# example main function. Assumes the existence of the file 'localdata.py'
# which contains train0 and test0 data list entries; you can just copy and rename
# the examples/data.py file

import exampledata

if __name__ == "__main__":
    training_data = exampledata.train0
    test_data = exampledata.test0
    learn_results_tuple = svm_learn( training_data, type='optimization' )

    write_model( learn_results_tuple.model, 'my_python_model.dat')

    predictions = svm_classify( learn_results_tuple.model, test_data)
    for p in predictions:
         print '%.8f' % p
    ''''
    # As things stand, this will fail because pickle cannot work
    # with Ctypes objects that use pointers
    with open("model.pickle", 'wb') as f:
        pickle.dump( learn_results_tuple.model, f)
    '''

# ----------------------------------------------


