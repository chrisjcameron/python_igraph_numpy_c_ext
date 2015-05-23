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


// Function signatures
static PyObject *ignp_fun_sumDegree(PyObject *self, PyObject *args);
static PyObject *ignp_fun_degreeList(PyObject *self, PyObject *args);
double sumDegree(const igraph_t *g);


// Create Module Def
static PyMethodDef module_methods[] = {
    {"sum_degree", ignp_fun_sumDegree,  METH_VARARGS, sumDegree_docstring},
    {"degree_array", ignp_fun_degreeList, METH_VARARGS, degreeList_docstring},
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

double sumDegree(const igraph_t *g) {
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
    igraph_t* g;
    double value;
    
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &x_obj))
        return NULL;

    g = (igraph_t*) PyCObject_AsVoidPtr(x_obj);

    /* Call the external C function to compute sum degree. */
    value = sumDegree(g);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
    
}

/* Array access macro
   Modeled on https://github.com/johnnylee/python-numpy-c-extension-examples */
#define deg(x0) (*(npy_int32*)((PyArray_DATA(py_deg) +                \
                                (x0) * PyArray_STRIDES(py_deg)[0])))

static PyObject *ignp_fun_degreeList( PyObject *self, PyObject *args ) {
    long int i;
    igraph_t *g;
    PyArrayObject *py_deg;
    PyObject *x_obj;
    igraph_vector_t v;
    igraph_vector_init(&v, 0);
    
    
    if (!PyArg_ParseTuple(args, "OO!",
                           &x_obj,
                           &PyArray_Type, &py_deg
                        )) {
        return NULL;
    }

    g = (igraph_t*) PyCObject_AsVoidPtr(x_obj);
    igraph_degree(g, &v, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
    
    for (i = 0; i<igraph_vector_size(&v); i++) {
        deg(i) = VECTOR(v)[ i ];
    }
    
    igraph_vector_destroy(&v);
    Py_RETURN_NONE;
} 

















