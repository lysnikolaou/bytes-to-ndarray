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

    if(!PyArg_ParseTuple(args, "O", &o)) {
        return NULL;
    }

    if (!PyList_Check(o)) {
        PyErr_SetString(PyExc_ValueError, "bytesarray only accepts a list of bytes");
        return NULL;
    }

    // Allocate buffer. Call calloc so that memory gets initialized to 0.
    Py_ssize_t size = PyList_Size(o);
    char **buffer = PyMem_Calloc(size, sizeof(char *));
    if (!buffer) {
        PyErr_NoMemory();
        return NULL;
    }

    // Allocate space for the sizes. We need these for copying over
    // from buffer to the array later.
    Py_ssize_t *sizes = PyMem_Malloc(size * sizeof(Py_ssize_t));
    if (!sizes) {
        PyErr_NoMemory();
        goto fail_after_buffer;
    }

    Py_ssize_t maxitemsize = 0;
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PyList_GetItem(o, i);
        if (!PyBytes_CheckExact(item)) {
            PyErr_SetString(PyExc_ValueError, "strarray only accepts a list of bytes");
            goto fail_after_sizes;
        }

        // Find maxsize. Needed for setting the dtype's elsize.
        Py_ssize_t itemsize = PyBytes_Size(item);
        sizes[i] = itemsize;
        if (itemsize > maxitemsize) {
            maxitemsize = itemsize;
        }

        buffer[i] = PyMem_Malloc((itemsize + 1) * sizeof(char));
        if (!buffer[i]) {
            PyErr_NoMemory();
            goto fail_after_sizes;
        }

        char *item_as_string = PyBytes_AsString(item);
        memcpy(buffer[i], item_as_string, itemsize + 1);
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_STRING);
    if (descr == NULL) {
        goto fail_after_sizes;
    }
    descr->elsize = maxitemsize; // Set buffer size for array items

    Py_INCREF(descr);
    PyArrayObject *arr = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, &size, descr);
    if (arr == NULL) {
        goto fail_after_descr;
    }

    // Create array iterator
    NpyIter *iter = NpyIter_New(arr, NPY_ITER_WRITEONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (!iter) {
        goto fail_after_descr;
    }

    // Save iterator's next() for more efficient calling in the while-loop
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (!iternext) {
        goto fail_after_iter;
    }

    // Get pointers to data, strides and inner loop size
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    Py_ssize_t i = 0;
    do {
        char *data = *dataptr;  // Data buffer
        npy_intp stride = *strideptr;  // Strides

        // Inner loop size (needed because of NPY_ITER_EXTERNAL_LOOP above)
        npy_intp count = *innersizeptr;

        while (count--) {
            memcpy(data, buffer[i], sizes[i]);

            // For elements where len(buffer[i]) < elsize, we need to set the
            // remaining bytes to 0.
            memset(data + sizes[i], 0, descr->elsize - sizes[i]);

            i++;
            data += stride;
        }

    } while (iternext(iter));


    res = (PyObject *) arr;

    // Clean everything up
fail_after_iter:
    NpyIter_Deallocate(iter);
fail_after_descr:
    Py_DECREF(descr);
fail_after_sizes:
    PyMem_Free(sizes);
fail_after_buffer:
    for (Py_ssize_t i = 0; i < size; i++) {
        if (!buffer[i]) {
            break;
        }
        PyMem_Free(buffer[i]);
    }
    PyMem_Free(buffer);
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
