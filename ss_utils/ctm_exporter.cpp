// ctm_exporter.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <openctm.h>  // Ensure your include path finds the OpenCTM headers

namespace py = pybind11;

void save_ctm(const std::string &filename, py::array_t<float> vertices, py::array_t<unsigned int> triangles) {
    // Request buffers from numpy arrays.
    auto buf_vertices = vertices.request();
    auto buf_triangles = triangles.request();

    if (buf_vertices.ndim != 2 || buf_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a 2D array with shape (n,3)");
    if (buf_triangles.ndim != 2 || buf_triangles.shape[1] != 3)
        throw std::runtime_error("triangles must be a 2D array with shape (m,3)");

    int num_vertices = buf_vertices.shape[0];
    int num_triangles = buf_triangles.shape[0];

    // Create a new OpenCTM export context.
    CTMcontext context = ctmNewContext(CTM_EXPORT);
    if (!context)
        throw std::runtime_error("Failed to create OpenCTM export context");

    // Define the mesh (we're not setting normals or additional attributes).
    ctmDefineMesh(context,
                  static_cast<float*>(buf_vertices.ptr),
                  num_vertices,
                  static_cast<unsigned int*>(buf_triangles.ptr),
                  num_triangles,
                  nullptr);

    // Save the file.
    // In this version, ctmSave returns void, so we don't check a return value.
    ctmSave(context, filename.c_str());

    // Clean up the context.
    ctmFreeContext(context);
}

PYBIND11_MODULE(ctm_exporter, m) {
    m.doc() = "CTM exporter module using the original OpenCTM package";
    m.def("save_ctm", &save_ctm, "Save a mesh to a CTM file",
          py::arg("filename"), py::arg("vertices"), py::arg("triangles"));
}

