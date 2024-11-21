#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <rclcpp/rclcpp.hpp>
#include "rclcpp/serialization.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <perception_interfaces/msg/feature_voxel_grid.hpp>
#include <string>
#include <cstdint>
#include <cstring>

typedef perception_interfaces::msg::FeatureVoxelGrid_< std::allocator < void > > FeatureVoxelGridMsg;

using namespace std::chrono_literals;

namespace py = pybind11;

void npcpp_voxel_grid(
    const py::array_t<float> np_features, 
    const py::array_t<uint64_t> np_indices,
    size_t features_addr,
    size_t indices_addr,
    const size_t & n_voxels, const size_t & n_features
) {
    float * features = reinterpret_cast<float *>(features_addr);
    uint64_t * indices = reinterpret_cast<uint64_t *>(indices_addr);

    std::memcpy(indices, np_indices.request().ptr, sizeof(uint64_t) * n_voxels);
    std::memcpy(features, np_features.request().ptr, sizeof(float) * n_voxels * n_features);

}


PYBIND11_MODULE(ros2_numpy_cpp, m) {
    m.def("npcpp_voxel_grid", 
        &npcpp_voxel_grid, 
        py::arg("np_features"),
        py::arg("np_indices"), 
        py::arg("features_addr"),
        py::arg("indices_addr"),
        py::arg("n_voxels"), py::arg("n_features"),
        "Copy data from numpy to cpp."
    );
}
