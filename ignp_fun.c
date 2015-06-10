#include <Python.h>
#include <numpy/arrayobject.h>
#include <igraph/igraph.h>



// Doc strings (can be NULL)
static char module_docstring[] =
    "This module used iGraph C library for fast custom computations.";
static char sumDegree_docstring[] =
    "Calculate the sum of node degrees in a given a graph.";
static char degreeList_docstring[] =
    "return node degrees in numpy list.";
static char propagate_docstring[] = 
    "propagate and compute stepwise results into passed arrays.";


// Function signatures
static PyObject *ignp_fun_sumDegree(PyObject *self, PyObject *args);
static PyObject *ignp_fun_degreeList(PyObject *self, PyObject *args);
static PyObject *ignp_fun_propagate(PyObject *self, PyObject *args);
static double sumDegree(const igraph_t *g);

// Create Module Def
static PyMethodDef module_methods[] = {
    {"sum_degree",   ignp_fun_sumDegree,  METH_VARARGS, sumDegree_docstring},
    {"degree_array", ignp_fun_degreeList, METH_VARARGS, degreeList_docstring},
    {"propagate",    ignp_fun_propagate,  METH_VARARGS, propagate_docstring},
    {NULL, NULL, 0, NULL}  /* sentinel to end table */
};

// Initialize Module
PyMODINIT_FUNC initignp_fun(void)
{
    PyObject *m = Py_InitModule3("ignp_fun", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();

}


/*****************************************************************************
 * Define Functions 
 ****************************************************************************/

static double sumDegree(const igraph_t *g) {
    int i;
    double result = 0.0;
    igraph_vector_t v;
    igraph_vector_init(&v, 0);
    igraph_degree(g, &v, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
    
    for (i = 0; i<igraph_vector_size(&v); i++) {
        result += VECTOR(v)[ (long int)i ];
    }
    igraph_vector_destroy(&v);
    return result;
}


static PyObject *ignp_fun_sumDegree(PyObject *self, PyObject *args)
{
    PyObject* x_obj;
    PyObject* mem_addr_o;
    long int mem_addr;
    igraph_t* g;
    double value;
    
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &x_obj))
        return NULL;
        
    mem_addr_o = PyObject_CallMethod(x_obj, "_raw_pointer", "()");
    mem_addr = PyInt_AsLong(mem_addr_o);
    Py_DECREF(mem_addr_o);
    if (mem_addr == -1) { 
        printf("PyInt to Long Failed");
        return NULL;
    }
    g = (igraph_t*) mem_addr;
    
    /* Call the external C function to compute sum degree. */
    value = sumDegree(g);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    PySys_WriteStdout("Finished Computing Sum\n");
    return ret;
    
}

/* Array access macro
   Modeled on https://github.com/johnnylee/python-numpy-c-extension-examples */
#define deg(x0) (*(npy_int64*)((PyArray_DATA(py_deg) +                \
                                (x0) * PyArray_STRIDES(py_deg)[0])))

static PyObject *ignp_fun_degreeList( PyObject *self, PyObject *args ) {
    long int i;
    igraph_t *g;
    PyArrayObject *py_deg;
    PyObject *x_obj;
    igraph_vector_t v;
    igraph_vector_init(&v, 0);
    PyObject* mem_addr_o;
    long int mem_addr;
    
    
    if (!PyArg_ParseTuple(args, "OO!",
                           &x_obj,
                           &PyArray_Type, &py_deg
                        )) {
        return NULL;
    }
    mem_addr_o = PyObject_CallMethod(x_obj, "_raw_pointer", "()");
    mem_addr = PyInt_AsLong(mem_addr_o);
    Py_DECREF(mem_addr_o);
    if (mem_addr == -1) { 
        printf("PyInt to Long Failed");
        return NULL;
    }
    g = (igraph_t*) mem_addr;

    igraph_degree(g, &v, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
    
    for (i = 0; i<igraph_vector_size(&v); i++) {
        deg(i) = VECTOR(v)[ i ];
    }
    
    igraph_vector_destroy(&v);
    Py_RETURN_NONE;
} 


/*---------------------------------
Propagate contagion on network
*/

/* Array access macro */
#define ax_i32(py_i32, x0) (*(npy_int32*)((PyArray_DATA(py_i32) +      \
                                (x0) * PyArray_STRIDES(py_i32)[0])))    

#define ax_i64(py_i64, x0) (*(npy_int64*)((PyArray_DATA(py_i64) +      \
                                (x0) * PyArray_STRIDES(py_i64)[0])))    

#define ax_f32(py_f32, x0) (*(npy_float32*)((PyArray_DATA(py_f32) +    \
                                (x0) * PyArray_STRIDES(py_f32)[0])))
                                
#define ax_i8(py_bool, x0) (*(npy_int8*)((PyArray_DATA(py_bool) +    \
                                (x0) * PyArray_STRIDES(py_bool)[0])))

static PyObject *ignp_fun_propagate(PyObject *self, PyObject *args) {
    long int num_active = 0;
    long int num_susc = 1;
    long int limit = 30;
    long int i;
    float lrAct;
    PyObject* mem_addr_o;
    long int mem_addr;
    
    /* StateTracker Vars */
    PyArrayObject *py_trkr; //   'i64'
    
    /* By EdgeID */
    PyArrayObject *py_tie_r; //  'f32'
    
    /* By NodeID */
    PyArrayObject *py_act_n;   // 'i8'
    PyArrayObject *py_thr_n;   // 'f32'
    PyArrayObject *py_exp_n;   // 'i64'

    /* By Infection Order*/
    PyArrayObject *py_deg;   //  i64
    PyArrayObject *py_nSuc;  //  i64
    PyArrayObject *py_nAct;  //  i64
    PyArrayObject *py_lrAct; //  f32
    PyArrayObject *py_hom;   //  i64
    PyArrayObject *py_eComp; //  i64
    PyArrayObject *py_iComp; //  i64
    PyArrayObject *py_eTri;  //  i64
    PyArrayObject *py_iTri;  //  i64
    PyArrayObject *py_thr;   //  i32
    PyArrayObject *py_exp;   //  i64
    PyArrayObject *py_cTime; //  i64

    PyObject *g_obj;
    igraph_t *g;
    igraph_t gc;
    
    long int randID;
    long int low  = 0;
    long int high = -1;
    long int ctime = 0;
    igraph_rng_t *rGen;
    igraph_vit_t nbr_iter;
    igraph_vs_t  nbr_sel;
    igraph_integer_t eid;
    igraph_integer_t vdeg;
    igraph_integer_t e_comp = 0;
    igraph_integer_t i_comp = 0;
    igraph_integer_t e_tri = 0;
    igraph_integer_t i_tri = 0;
    int actv_nbr_count;
    //int res, j;
    igraph_vector_t temp;
    //igraph_vector_t actv_nbrs;

    //PySys_WriteStdout("Parse Started\n");
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!",
                           &g_obj,
                           &PyArray_Type, &py_trkr,  //  i64
                           &PyArray_Type, &py_tie_r, //  'f32'
                           &PyArray_Type, &py_act_n, //  'i8'
                           &PyArray_Type, &py_thr_n, //  'i32'
                           &PyArray_Type, &py_exp_n, //  'i64'
                           &PyArray_Type, &py_deg,   //  i64
                           &PyArray_Type, &py_nSuc,  //  i64
                           &PyArray_Type, &py_nAct,  //  i64
                           &PyArray_Type, &py_lrAct, //  f32
                           &PyArray_Type, &py_hom,   //  i64
                           &PyArray_Type, &py_eComp, //  i64
                           &PyArray_Type, &py_iComp, //  i64
                           &PyArray_Type, &py_eTri,  //  i64
                           &PyArray_Type, &py_iTri,  //  i64
                           &PyArray_Type, &py_thr,   //  i64
                           &PyArray_Type, &py_exp,   //  i64 
                           &PyArray_Type, &py_cTime  //  i64                           
                        )) {
        printf("Parse Failed\n");
        Py_RETURN_NONE;
    }
    //PySys_WriteStdout("Getting Tracker Vars\n");
    num_active =  (long) ax_i64(py_trkr, 0);
    num_susc   =  (long) ax_i64(py_trkr, 1);
    limit      =  (long) ax_i64(py_trkr, 2);
    
    mem_addr_o = PyObject_CallMethod(g_obj, "_raw_pointer", "()");
    mem_addr = PyInt_AsLong(mem_addr_o);
    Py_DECREF(mem_addr_o);
    if (mem_addr == -1) { 
        printf("PyInt to Long Failed");
        return NULL;
    }
    g = (igraph_t*) mem_addr;
    
    //Setup Vars
    rGen = igraph_rng_default();
    //igraph_rng_init(rGen, time(NULL));
    high += (long) igraph_vcount(g);
    
    //PySys_WriteStdout("Propagate Starting with %li active of target %li with %li open\n",
    //                    num_active, limit, num_susc);
    //Propagate
    do {
        // get random node
        ctime += 1;
        randID = igraph_rng_get_integer(rGen, low, high);
        if ( ax_i8(py_act_n, randID) != 1 && ax_i64(py_exp_n, randID)>=ax_i32(py_thr_n, randID) ){
            //activate
            ax_i8(py_act_n,randID) = 1;
            lrAct = 0;
            
            //update nbrs
            actv_nbr_count = 0;
            igraph_vs_adj( &nbr_sel, randID, IGRAPH_ALL);
            igraph_vit_create(g, nbr_sel, &nbr_iter);
            igraph_vs_size(g, &nbr_sel, &vdeg);
            igraph_vector_init(&temp, vdeg);
            while( !IGRAPH_VIT_END(nbr_iter) ){
                i = (long int) IGRAPH_VIT_GET(nbr_iter);
                ax_i64( py_exp_n, i ) += 1;
                
                /* update active nbr count and collect id of active */
                if ( ax_i8(py_act_n, i) == i ) {
                    VECTOR(temp)[actv_nbr_count]=i;
                    actv_nbr_count += 1;
                }
                
                /* update num_susc */
                if ( ax_i8(py_act_n, i) == 0 && \
                     ax_i32(py_thr_n, i) >  (float) (ax_i64(py_exp_n, i)-1) && \
                     ax_i32(py_thr_n, i) <= (float) ax_i64(py_exp_n, i)      ){
                     /*PySys_WriteStdout("%li <  %i <= %li\n", 
                                         (ax_i64(py_exp_n, i)-1),
                                         ax_i32(py_thr_n, i),
                                         ax_i64(py_exp_n, i) );*/
                     num_susc += 1; 
                }
                
                /* Get #active long ties */
                if ( ax_i8(py_act_n, i) == 1 ){
                    igraph_get_eid(g, &eid, randID, i, 0, 1);
                    lrAct +=  ax_f32( py_tie_r, eid )>2 ;
                }
                IGRAPH_VIT_NEXT(nbr_iter);
            }
            igraph_vit_destroy(&nbr_iter);
            igraph_vs_destroy(&nbr_sel);

            //Compute Components (among all and active nbrs)
            igraph_vs_adj( &nbr_sel, randID, IGRAPH_ALL);
            igraph_induced_subgraph(g, &gc, nbr_sel, IGRAPH_SUBGRAPH_CREATE_FROM_SCRATCH);                         
            igraph_clusters(&gc, NULL, NULL, &e_comp, IGRAPH_WEAK);
            e_tri = igraph_vcount(&gc);
            igraph_destroy(&gc);
            igraph_vs_destroy(&nbr_sel);

        
            igraph_induced_subgraph(g, &gc, igraph_vss_vector(&temp), \
                                     IGRAPH_SUBGRAPH_CREATE_FROM_SCRATCH);
            igraph_clusters(&gc, NULL, NULL, &i_comp, IGRAPH_WEAK);
            i_tri = igraph_vcount(&gc);
            
            //Clean up
            igraph_destroy(&gc); 
            igraph_vector_destroy(&temp);
            
            //PySys_WriteStdout("e_comp: %i,  i_comp: %i\n", e_comp, i_comp);
            //PySys_WriteStdout("e_tri:  %i,  i_tri:  %i\n", e_tri, i_tri);

            //update tracking vars
            ax_f32( py_lrAct, num_active ) = (npy_float32) lrAct;
            ax_i32( py_thr,   num_active)  =  ax_i32(py_thr_n, randID);

            ax_i64( py_deg,   num_active) = (npy_int64) vdeg; 
            ax_i64( py_nSuc,  num_active) = (npy_int64) num_susc;
            ax_i64( py_nAct,  num_active) = (npy_int64) num_active;
            //ax_i64( py_hom,   num_active) = (npy_int64) num_susc;
            ax_i64( py_eComp, num_active) = (npy_int64) e_comp;
            ax_i64( py_iComp, num_active) = (npy_int64) i_comp;
            ax_i64( py_eTri,  num_active) = (npy_int64) e_tri;
            ax_i64( py_iTri,  num_active) = (npy_int64) i_tri;
            ax_i64( py_exp,   num_active) = ax_i64(py_exp_n, randID);
            ax_i64( py_cTime, num_active) = (npy_int64) ctime;
            num_active += 1;
        }
    } while( num_susc > num_active && num_active < limit);
    //PySys_WriteStdout("Propagate Finished with %li active of target %li with %li open\n",
    //                   num_active, limit, num_susc);

    //igraph_rng_destroy(rGen);
    ax_i64(py_trkr, 0) = (npy_int64) num_active;
    ax_i64(py_trkr, 1) = (npy_int64) num_susc  ;
    ax_i64(py_trkr, 2) = (npy_int64) limit     ;

    Py_RETURN_NONE;

}




