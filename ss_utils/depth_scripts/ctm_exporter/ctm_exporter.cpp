#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <openctm.h>   // Ensure your include path finds the OpenCTM headers
#include <limits>
#include <algorithm>

namespace py = pybind11;

void save_ctm(const std::string &filename, py::array_t<float> vertices, py::array_t<int> triangles, 
              py::object user_offset = py::none(),
              double vertex_prec = 0.001, 
              int compression = 9) {
    // Request buffers from numpy arrays.
    auto buf_vertices = vertices.request();
    auto buf_triangles = triangles.request();

    if (buf_vertices.ndim != 2 || buf_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a 2D array with shape (n,3)");
    if (buf_triangles.ndim != 2 || buf_triangles.shape[1] != 3)
        throw std::runtime_error("triangles must be a 2D array with shape (m,3)");

    int num_vertices = buf_vertices.shape[0];
    int num_triangles = buf_triangles.shape[0];

    // Determine the bounding box of the mesh to calculate center
    const float* vertices_ptr = static_cast<const float*>(buf_vertices.ptr);
    
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    
    for (int i = 0; i < num_vertices; ++i) {
        float x = vertices_ptr[i * 3];
        float y = vertices_ptr[i * 3 + 1];
        float z = vertices_ptr[i * 3 + 2];
        
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
    }
    
    // Calculate center of mesh in x and y
    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    
    // Use the minimum z value as the base (or override with user value)
    double center_z = min_z;
    
    // Determine the offset to center the mesh at (0,0) for X and Y
    double offset_x, offset_y, offset_z;
    
    if (!user_offset.is_none()) {
        // User specified an offset, use it
        py::tuple offset_tuple = py::cast<py::tuple>(user_offset);
        if (py::len(offset_tuple) != 3) {
            throw std::runtime_error("Offset must be a tuple with 3 elements (x, y, z)");
        }
        
        offset_x = py::cast<double>(offset_tuple[0]);
        offset_y = py::cast<double>(offset_tuple[1]);
        offset_z = py::cast<double>(offset_tuple[2]);
    } else {
        // Auto-calculate offset to center the mesh at (0,0) in X and Y
        offset_x = center_x;
        offset_y = center_y;
        offset_z = center_z;
    }
    
    // Create a new OpenCTM export context.
    CTMcontext context = ctmNewContext(CTM_EXPORT);
    if (!context) {
        throw std::runtime_error("Failed to create OpenCTM export context");
    }

    // Extract vertices and apply the offset to center the mesh
    std::vector<float> vertices_float;
    vertices_float.reserve(num_vertices * 3);
    
    for (int i = 0; i < num_vertices; ++i) {
        vertices_float.push_back(vertices_ptr[i * 3] - static_cast<float>(offset_x));
        vertices_float.push_back(vertices_ptr[i * 3 + 1] - static_cast<float>(offset_y));
        vertices_float.push_back(vertices_ptr[i * 3 + 2] - static_cast<float>(offset_z));
    }

    // Define the mesh
    ctmDefineMesh(context,
                  vertices_float.data(),
                  num_vertices,
                  static_cast<const CTMuint*>(buf_triangles.ptr),
                  num_triangles,
                  nullptr);

    // Format offset as a string exactly as in the ReadOpenCtmFileFromUrl function
    char offset_string[256];
    snprintf(offset_string, sizeof(offset_string), "%.8f;%.8f;%.8f", offset_x, offset_y, offset_z);
    
    // Set the file comment with the offset string
    ctmFileComment(context, offset_string);

    // Set compression method to MG2 (same as in WriteOpenCtmFileToUrlEx)
    ctmCompressionMethod(context, CTM_METHOD_MG2);
    
    // Set vertex precision
    ctmVertexPrecision(context, static_cast<float>(vertex_prec));
    
    // Set compression level
    ctmCompressionLevel(context, compression);

    // Save the file
    ctmSave(context, filename.c_str());

    // Clean up the context
    ctmFreeContext(context);
}

PYBIND11_MODULE(ctm_exporter, m) {
    m.doc() = "CTM exporter module compatible with ReadOpenCtmFileFromUrl with auto-centering";
    m.def("save_ctm", &save_ctm, "Save a mesh to a CTM file with automatic x,y centering",
          py::arg("filename"), py::arg("vertices"), py::arg("triangles"), 
          py::arg("user_offset") = py::none(),
          py::arg("vertex_prec") = 0.001,
          py::arg("compression") = 9);
}