#include "pybind11_tests.h"

#include <pybind11/pybind11.h>
#include <pybind11/dlpack_tensor.h>
#include <algorithm>

namespace py = pybind11;

using namespace py::literals;

int destruct_count = 0;

TEST_MODULE(test_tensor_ext, m) {
    m.def("get_shape", [](const py::tensor<> &t) {
        py::list l;
        for (size_t i = 0; i < t.ndim(); ++i)
            l.append(t.shape(i));
        return l;
    }, "array"_a.noconvert());

    m.def("check_float", [](const py::tensor<> &t) {
        return t.dtype() == py::dtype<float>();
    });

    m.def("pass_float32", [](const py::tensor<float> &) { }, "array"_a.noconvert());
    m.def("pass_uint32", [](const py::tensor<uint32_t> &) { }, "array"_a.noconvert());
    m.def("pass_float32_shaped",
          [](const py::tensor<float, py::shape<3, py::any, 4>> &) {}, "array"_a.noconvert());

    m.def("pass_float32_shaped_ordered",
          [](const py::tensor<float, py::c_contig,
                              py::shape<py::any, py::any, 4>> &) {}, "array"_a.noconvert());

    m.def("check_order", [](py::tensor<py::c_contig>) -> char { return 'C'; });
    m.def("check_order", [](py::tensor<py::f_contig>) -> char { return 'F'; });
    m.def("check_order", [](py::tensor<>) -> char { return '?'; });

    m.def("check_device", [](py::tensor<py::device::cpu>) -> const char * { return "cpu"; });
    m.def("check_device", [](py::tensor<py::device::cuda>) -> const char * { return "cuda"; });

    m.def("initialize",
          [](py::tensor<float, py::shape<10>, py::device::cpu> &t) {
              for (size_t i = 0; i < 10; ++i)
                t(i) = (float) i;
          });

    m.def("initialize",
          [](py::tensor<float, py::shape<10, py::any>, py::device::cpu> &t) {
              int k = 0;
              for (size_t i = 0; i < 10; ++i)
                  for (size_t j = 0; j < t.shape(1); ++j)
                      t(i, j) = (float) k++;
          });

    m.def(
        "noimplicit",
        [](py::tensor<float, py::c_contig, py::shape<2, 2>>) { return 0; },
        "array"_a.noconvert());

    m.def(
        "implicit",
        [](py::tensor<float, py::c_contig, py::shape<2, 2>>) { return 0; },
        "array"_a);

    m.def("inspect_tensor", [](py::tensor<> tensor) {
        printf("Tensor data pointer : %p\n", tensor.data());
        printf("Tensor dimension : %zu\n", tensor.ndim());
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            printf("Tensor dimension [%zu] : %zu\n", i, tensor.shape(i));
            printf("Tensor stride    [%zu] : %zu\n", i, (size_t) tensor.stride(i));
        }
        printf("Tensor is on CPU? %i\n", tensor.device_type() == py::device::cpu::value);
        printf("Device ID = %u\n", tensor.device_id());
        printf("Tensor dtype check: int16=%i, uint32=%i, float32=%i\n",
            tensor.dtype() == py::dtype<int16_t>(),
            tensor.dtype() == py::dtype<uint32_t>(),
            tensor.dtype() == py::dtype<float>()
        );
    });

    m.def("process", [](py::tensor<uint8_t, py::shape<py::any, py::any, 3>,
                                   py::c_contig, py::device::cpu> tensor) {
        // Double brightness of the MxNx3 RGB image
        for (size_t y = 0; y < tensor.shape(0); ++y)
            for (size_t x = 0; y < tensor.shape(1); ++x)
                for (size_t ch = 0; ch < 3; ++ch)
                    tensor(y, x, ch) = (uint8_t) std::min(255, tensor(y, x, ch) * 2);

    });

    m.def("destruct_count", []() { return destruct_count; });
    m.def("return_dlpack", []() {
        float *f = new float[8] { 1, 2, 3, 4, 5, 6, 7, 8 };
        size_t shape[2] = { 2, 4 };

        py::capsule deleter(f, [](void *data) noexcept {
            destruct_count++;
            delete[] (float *) data;
        });

        return py::tensor<float, py::shape<2, 4>>(f, 2, shape, deleter);
    });
    m.def("passthrough", [](py::tensor<> a) { return a; });

    m.def("ret_numpy", []() {
        float *f = new float[8] { 1, 2, 3, 4, 5, 6, 7, 8 };
        size_t shape[2] = { 2, 4 };

        py::capsule deleter(f, [](void *data) noexcept {
            destruct_count++;
            delete[] (float *) data;
        });

        return py::tensor<py::numpy, float, py::shape<2, 4>>(f, 2, shape,
                                                             deleter);
    });

    m.def("ret_pytorch", []() {
        float *f = new float[8] { 1, 2, 3, 4, 5, 6, 7, 8 };
        size_t shape[2] = { 2, 4 };

        py::capsule deleter(f, [](void *data) noexcept {
           destruct_count++;
           delete[] (float *) data;
        });

        return py::tensor<py::pytorch, float, py::shape<2, 4>>(f, 2, shape,
                                                               deleter);
    });
}
