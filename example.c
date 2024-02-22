#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdlib.h>

static PyObject *example_bytesarray(PyObject *self, PyObject *args);

static PyMethodDef ExampleMethods[] = {
    {"bytesarray", example_bytesarray, METH_VARARGS, "Create a NumPy array containing bytes"},
    {NULL, NULL, 0, NULL},
};

static PyObject *example_bytesarray(PyObject *self, PyObject *args)
{
    PyObject *res = NULL;
    PyObject *o;

    /* This parses the Python argument into a double */
    if(!PyArg_ParseTuple(args, "O", &o)) {
        return NULL;
    }

    if (!PyList_Check(o)) {
        PyErr_SetString(PyExc_ValueError, "strarray only accepts a list of bytes");
        return NULL;
    }

    Py_ssize_t size = PyList_Size(o);
    char **buffer = PyMem_Calloc(size, sizeof(char *));
    if (!buffer) {
        PyErr_NoMemory();
        return NULL;
    }

    Py_ssize_t *sizes = PyMem_Malloc(size * sizeof(Py_ssize_t));
    if (!sizes) {
        PyErr_NoMemory();
        PyMem_Free(buffer);
        return NULL;
    }

    Py_ssize_t maxitemsize = 0;
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PyList_GetItem(o, i);
        if (!PyBytes_CheckExact(item)) {
            PyErr_SetString(PyExc_ValueError, "strarray only accepts a list of bytes");
            goto fail;
        }

        Py_ssize_t itemsize = PyBytes_Size(item);
        sizes[i] = itemsize;
        if (itemsize > maxitemsize) {
            maxitemsize = itemsize;
        }

        buffer[i] = PyMem_Malloc((itemsize + 1) * sizeof(char));
        if (!buffer[i]) {
            PyErr_NoMemory();
            goto fail;
        }

        char *item_as_string = PyBytes_AsString(item);
        memcpy(buffer[i], item_as_string, itemsize + 1);
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_STRING);
    if (descr == NULL) {
        goto fail;
    }
    descr->elsize = maxitemsize;

    Py_INCREF(descr);
    PyArrayObject *arr = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, &size, descr);
    if (arr == NULL) {
        Py_DECREF(descr);
        goto fail;
    }

    NpyIter *iter = NpyIter_New(arr, NPY_ITER_WRITEONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (!iter) {
        Py_DECREF(descr);
        goto fail;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (!iternext) {
        Py_DECREF(descr);
        NpyIter_Deallocate(iter);
        goto fail;
    }

    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    Py_ssize_t i = 0;
    do {
        char *data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            memcpy(data, buffer[i], sizes[i]);
            memset(buffer + sizes[i], 0, descr->elsize - sizes[i]);
            i++;
            data += stride;
        }

    } while (iternext(iter));


    Py_DECREF(descr);
    NpyIter_Deallocate(iter);
    res = (PyObject *) arr;

fail:
    for (Py_ssize_t i = 0; i < size; i++) {
        if (!buffer[i]) {
            break;
        }
        PyMem_Free(buffer[i]);
    }
    PyMem_Free(buffer);
    PyMem_Free(sizes);
    return res;
}


/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "example",
    NULL,
    -1,
    ExampleMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_example(void)
{
    import_array();

    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
