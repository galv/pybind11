/*
    pybind11/dlpack.h: Transparent conversion for tensors convertible to DLPack format
    Copyright (c) 2022 NVIDIA
    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "detail/common.h"

#include <atomic>
#include <deque>
#include <list>
#include <map>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <valarray>
#include <type_traits>

#include "dlpack.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

constexpr size_t any = (size_t) -1;

template <size_t... Is> struct shape {
    static constexpr size_t size = sizeof...(Is);
};

struct c_contig { };
struct f_contig { };
struct numpy { };
struct tensorflow { };
struct pytorch { };
struct jax { };

template <typename T> constexpr DLDataType dtype() {
    static_assert(
        std::is_floating_point<T>::value || std::is_integral<T>::value,
        "pybind11::dtype<T>: T must be a floating point or integer variable!"
    );

    DLDataType result;

    if (std::is_floating_point<T>::value)
        result.code = kDLFloat;
    else if (std::is_signed<T>::value)
        result.code = kDLInt;
    else
        result.code = kDLUInt;

    result.bits = sizeof(T) * 8;
    result.lanes = 1;

    return result;
}

PYBIND11_NAMESPACE_BEGIN(detail)

bool DLDataType_equal(const DLDataType& d1, const DLDataType& d2) {
    return d1.code == d2.code && d1.bits == d2.bits && d1.lanes == d2.lanes;
}

template <typename T> struct scoped_pymalloc {
    scoped_pymalloc(size_t size = 1) {
        ptr = (T *) PyMem_Malloc(size * sizeof(T));
        if (!ptr)
            assert(0 && "scoped_pymalloc(): could not allocate %zu bytes of memory!", size);
    }
    ~scoped_pymalloc() { PyMem_Free(ptr); }
    T *release() {
        T *temp = ptr;
        ptr = nullptr;
        return temp;
    }
    T *get() const { return ptr; }
    T &operator[](size_t i) { return ptr[i]; }
    T *operator->() { return ptr; }
private:
    T *ptr{ nullptr };
};

enum class tensor_framework : int { none, numpy, tensorflow, pytorch, jax };

struct tensor_req {
    DLDataType dtype;
    uint32_t ndim = 0;
    size_t *shape = nullptr;
    bool req_shape = false;
    bool req_dtype = false;
    char req_order = '\0';
    uint8_t req_device = 0;
};

template <typename T, typename = int> struct tensor_arg {
    static constexpr size_t size = 0;
    static constexpr auto name = descr<0>{ };
    static void apply(tensor_req &) { }
};

template <typename T> struct tensor_arg<T, enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr size_t size = 0;

    static constexpr auto name =
        const_name("dtype=float") + const_name<sizeof(T) * 8>();

    static void apply(tensor_req &tr) {
        tr.dtype = dtype<T>();
        tr.req_dtype = true;
    }
};

template <typename T> struct tensor_arg<T, enable_if_t<std::is_integral<T>::value>> {
    static constexpr size_t size = 0;

    static constexpr auto name =
        const_name("dtype=") + const_name<std::is_unsigned<T>::value>("u", "") +
        const_name("int") + const_name<sizeof(T) * 8>();

    static void apply(tensor_req &tr) {
        tr.dtype = dtype<T>();
        tr.req_dtype = true;
    }
};

template <size_t... Is> struct tensor_arg<shape<Is...>> {
    static constexpr size_t size = sizeof...(Is);
    static constexpr auto name =
        const_name("shape=(") +
        concat(const_name<Is == any>(const_name("*"), const_name<Is>())...) +
        const_name(")");

    static void apply(tensor_req &tr) {
        size_t i = 0;
        ((tr.shape[i++] = Is), ...);
        tr.ndim = (uint32_t) sizeof...(Is);
        tr.req_shape = true;
    }
};

template <> struct tensor_arg<c_contig> {
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("order='C'");
    static void apply(tensor_req &tr) { tr.req_order = 'C'; }
};

template <> struct tensor_arg<f_contig> {
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("order='F'");
    static void apply(tensor_req &tr) { tr.req_order = 'F'; }
};

template <typename T> struct tensor_arg<T, enable_if_t<T::is_device>> {
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("device='") + T::name + const_name("'");
    static void apply(tensor_req &tr) { tr.req_device = (uint8_t) T::value; }
};

template <typename... Ts> struct tensor_info {
    using scalar_type = void;
    using shape_type = void;
    constexpr static auto name = const_name("tensor");
    constexpr static tensor_framework framework = tensor_framework::none;
};

template <typename T, typename... Ts> struct tensor_info<T, Ts...>  : tensor_info<Ts...> {
    using scalar_type =
        std::conditional_t<std::is_scalar<T>::value, T,
                           typename tensor_info<Ts...>::scalar_type>;
};

template <size_t... Is, typename... Ts> struct tensor_info<shape<Is...>, Ts...> : tensor_info<Ts...> {
    using shape_type = shape<Is...>;
};

template <typename... Ts> struct tensor_info<numpy, Ts...> : tensor_info<Ts...> {
    constexpr static auto name = const_name("numpy.ndarray");
    constexpr static tensor_framework framework = tensor_framework::numpy;
};

template <typename... Ts> struct tensor_info<pytorch, Ts...> : tensor_info<Ts...> {
    constexpr static auto name = const_name("torch.Tensor");
    constexpr static tensor_framework framework = tensor_framework::pytorch;
};

template <typename... Ts> struct tensor_info<tensorflow, Ts...> : tensor_info<Ts...> {
    constexpr static auto name = const_name("tensorflow.python.framework.ops.EagerTensor");
    constexpr static tensor_framework framework = tensor_framework::tensorflow;
};

template <typename... Ts> struct tensor_info<jax, Ts...> : tensor_info<Ts...> {
    constexpr static auto name = const_name("jaxlib.xla_extension.DeviceArray");
    constexpr static tensor_framework framework = tensor_framework::jax;
};


struct tensor_handle {
    DLManagedTensor *tensor;
    std::atomic<size_t> refcount;
    PyObject *owner;
    bool free_shape;
    bool free_strides;
    bool call_deleter;
};

/// Increase the reference count of the given tensor object; returns a pointer
/// to the underlying DLtensor
DLTensor *tensor_inc_ref(tensor_handle *th) noexcept {
    if (!th)
        return nullptr;
    ++th->refcount;
    return &th->tensor->dl_tensor;
}

/// Decrease the reference count of the given tensor object
void tensor_dec_ref(tensor_handle *th) noexcept {
    if (!th)
        return;
    size_t rc_value = th->refcount--;

    if (rc_value == 0) {
        // TODO: better altenrative to nanobind::fail
        assert(0 && "tensor_dec_ref(): reference count became negative!");
    } else if (rc_value == 1) {
        Py_XDECREF(th->owner);
        DLManagedTensor *mt = th->tensor;
        if (th->free_shape) {
            PyMem_Free(mt->dl_tensor.shape);
            mt->dl_tensor.shape = nullptr;
        }
        if (th->free_strides) {
            PyMem_Free(mt->dl_tensor.strides);
            mt->dl_tensor.strides = nullptr;
        }
        if (th->call_deleter) {
            if (mt->deleter)
                mt->deleter(mt);
        } else {
            PyMem_Free(mt);
        }
        PyMem_Free(th);
    }
}

static void tensor_capsule_destructor(PyObject *o) {
    error_scope scope; // temporarily save any existing errors
    DLManagedTensor *mt =
        (DLManagedTensor *) PyCapsule_GetPointer(o, "dltensor");
    if (mt)
        tensor_dec_ref((tensor_handle *) mt->manager_ctx);
    else
        PyErr_Clear();
}

/// Wrap a tensor_handle* into a PyCapsule
PyObject *tensor_wrap(tensor_handle *th, int framework) noexcept {
    tensor_inc_ref(th);
    object o = reinterpret_steal<object>(PyCapsule_New(th->tensor, "dltensor", tensor_capsule_destructor));
    // object package;

    // switch ((tensor_framework) framework) {
    //     case tensor_framework::none:
    //         break;

    //     case tensor_framework::numpy:
    //         package = module_::import_("numpy");
    //         // I don't understand this? handle from PyTypeObject? I see...
    //         // Why is numpy weird about this?
    //         // o = handle(internals_get().nb_tensor)(o);
    //         break;

    //     case tensor_framework::pytorch:
    //         package = module_::import_("torch.utils.dlpack");
    //         break;


    //     case tensor_framework::tensorflow:
    //         package = module_::import_("tensorflow.experimental.dlpack");
    //         break;

    //     case tensor_framework::jax:
    //         package = module_::import_("jax.dlpack");
    //         break;


    //     default:
    //         fail("pybind11::detail::tensor_wrap(): unknown framework "
    //              "specified!");
    // }

    // // package matters only for numpy framework. Why?
    // if (package.is_valid()) {
    //     try {
    //         o = package.attr("from_dlpack")(o);
    //     } catch (...) {
    //         if ((tensor_framework) framework == tensor_framework::numpy) {
    //             try {
    //                 // Older numpy versions
    //                 o = package.attr("_from_dlpack")(o);
    //             } catch (...) {
    //                 try {
    //                     // Yet older numpy versions
    //                     o = package.attr("asarray")(o);
    //                 } catch (...) {
    //                     return nullptr;
    //                 }
    //             }
    //         } else {
    //             return nullptr;
    //         }
    //     }
    // }

    return o.release().ptr();
}

tensor_handle *tensor_import(PyObject *o, const tensor_req *req,
                             bool convert) noexcept {
    object capsule;

    // If this is not a capsule, try calling o.__dlpack__()
    if (!PyCapsule_CheckExact(o)) {
        capsule = reinterpret_steal<object>(PyObject_CallMethod(o, "__dlpack__", nullptr));

        if (capsule.ptr() == nullptr) {
            PyErr_Clear();
            PyObject *tp = (PyObject*) Py_TYPE(o);

            try {
                str module_object = reinterpret_steal<str>(PyObject_GetAttrString(tp, "__module__"));
                const char *module_name = PyUnicode_AsUTF8AndSize(module_object.ptr(), nullptr);

                object package;
                if (strncmp(module_name, "tensorflow.", 11) == 0)
                    package = module_::import("tensorflow.experimental.dlpack");
                else if (strcmp(module_name, "torch") == 0)
                    package = module_::import("torch.utils.dlpack");
                else if (strncmp(module_name, "jaxlib", 6) == 0)
                    package = module_::import("jax.dlpack");

                if (package.ptr() != nullptr)
                    capsule = package.attr("to_dlpack")(handle(o));
            } catch (...) {
                // clear() isn't defined in pybind11, but nevertheless it's not clear to me 
                // that this is necessary anyway.
                // capsule.clear();
            }
        }

        // Try creating a tensor via the buffer protocol
        // TODO: Consider removing this for now. Not sure what buffer protocol is
        // if (!capsule.is_valid())
        //     capsule = reinterpet_steal<object>(dlpack_from_buffer_protocol(o));

        if (capsule == nullptr)
            return nullptr;
    } else {
        capsule = reinterpret_borrow<object>(o);
    }

    // Extract the pointer underlying the capsule
    void *ptr = PyCapsule_GetPointer(capsule.ptr(), "dltensor");
    if (!ptr) {
        PyErr_Clear();
        return nullptr;
    }

    // Check if the tensor satisfies the requirements
    DLTensor &t = ((DLManagedTensor *) ptr)->dl_tensor;

    bool pass_dtype = true, pass_device = true,
         pass_shape = true, pass_order = true;

    if (req->req_dtype)
        pass_dtype = DLDataType_equal(t.dtype, req->dtype);

    if (req->req_device)
        pass_device = t.device.device_type == req->req_device;

    if (req->req_shape) {
        pass_shape &= req->ndim == (uint32_t) t.ndim;

        if (pass_shape) {
            for (uint32_t i = 0; i < req->ndim; ++i) {
                if (req->shape[i] != (size_t) t.shape[i] &&
                    req->shape[i] != pybind11::any) {
                    pass_shape = false;
                    break;
                }
            }
        }
    }

    scoped_pymalloc<int64_t> strides(t.ndim);
    if ((req->req_order || t.strides == nullptr) && t.ndim > 0) {
        size_t accum = 1;

        if (req->req_order == 'C' || t.strides == nullptr) {
            for (uint32_t i = (uint32_t) (t.ndim - 1);;) {
                strides[i] = accum;
                accum *= t.shape[i];
                if (i == 0)
                    break;
                --i;
            }
        } else if (req->req_order == 'F') {
            pass_order &= t.strides != nullptr;

            for (uint32_t i = 0; i < (uint32_t) t.ndim; ++i) {
                strides[i] = accum;
                accum *= t.shape[i];
            }
        } else {
            pass_order = false;
        }

        if (t.strides) {
            for (uint32_t i = 0; i < (uint32_t) t.ndim; ++i) {
                if (strides[i] != t.strides[i]) {
                    pass_order = false;
                    break;
                }
            }
        }
    }

    // Support implicit conversion of 'dtype' and order
    if (pass_device && pass_shape && (!pass_dtype || !pass_order) && convert &&
        capsule.ptr() != o) {
        PyTypeObject *tp = Py_TYPE(o);
        str module_name_o = reinterpret_borrow<str>(handle(tp).attr("__module__"));
        const std::string module_name = module_name_o.str();

        char order = 'K';
        if (req->req_order != '\0')
            order = req->req_order;

        if (req->dtype.lanes != 1)
            return nullptr;

        const char *prefix = nullptr;
        char dtype[8];
        switch (req->dtype.code) {
            case (uint8_t) pybind11::dtype_code::Int: prefix = "int"; break;
            case (uint8_t) pybind11::dtype_code::UInt: prefix = "uint"; break;
            case (uint8_t) pybind11::dtype_code::Float: prefix = "float"; break;
            default:
                return nullptr;
        }
        snprintf(dtype, sizeof(dtype), "%s%u", prefix, req->dtype.bits);

        object converted;
        try {
            if (strcmp(module_name.c_str(), "numpy") == 0) {
                converted = handle(o).attr("astype")(dtype, order);
            } else if (strcmp(module_name.c_str(), "torch") == 0) {
                converted = handle(o).attr("to")(
                    arg("dtype") = module_::import_("torch").attr(dtype),
                    arg("copy") = true
                );
            } else if (strncmp(module_name.c_str(), "tensorflow.", 11) == 0) {
                converted = module_::import_("tensorflow")
                                .attr("cast")(handle(o), dtype);
            } else if (strncmp(module_name.c_str(), "jaxlib", 6) == 0) {
                converted = handle(o).attr("astype")(dtype);
            }
        } catch (...) { converted.clear(); }

        // Potentially try again recursively
        if (!converted.is_valid())
            return nullptr;
        else
            return tensor_import(converted.ptr(), req, false);
    }

    if (!pass_dtype || !pass_device || !pass_shape || !pass_order)
        return nullptr;

    // Create a reference-counted wrapper
    scoped_pymalloc<tensor_handle> result;
    result->tensor = (DLManagedTensor *) ptr;
    result->refcount = 0;
    result->owner = nullptr;
    result->free_shape = false;
    result->call_deleter = true;

    // Ensure that the strides member is always initialized
    if (t.strides) {
        result->free_strides = false;
    } else {
        result->free_strides = true;
        t.strides = strides.release();
    }

    // Mark the dltensor capsule as "consumed"
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") ||
        PyCapsule_SetDestructor(capsule.ptr(), nullptr))
        assert(0 && "nanobind::detail::tensor_import(): could not mark dltensor "
               "capsule as consumed!");

    return result.release();
}

PYBIND11_NAMESPACE_END(detail)

template <typename... Args> class tensor {
public:
    using Info = detail::tensor_info<Args...>;
    using Scalar = typename Info::scalar_type;

    tensor() = default;

    explicit tensor(detail::tensor_handle *handle) : m_handle(handle) {
        if (handle)
            m_tensor = *detail::tensor_inc_ref(handle);
    }

    tensor(void *value,
           size_t ndim,
           const size_t *shape,
           handle owner = pybind11::handle(),
           const int64_t *strides = nullptr,
           DLDataType dtype = pybind11::datatype<Scalar>(),
           int32_t device_type = device::cpu::value,
           int32_t device_id = 0) {
        m_handle = detail::tensor_create(value, ndim, shape, owner.ptr(), strides,
                                         &dtype, device_type, device_id);
        m_tensor = *detail::tensor_inc_ref(m_handle);
    }

    ~tensor() {
        detail::tensor_dec_ref(m_handle);
    }

    tensor(const tensor &t) : m_handle(t.m_handle), m_tensor(t.m_tensor) {
        detail::tensor_inc_ref(m_handle);
    }

    tensor(tensor &&t) noexcept : m_handle(t.m_handle), m_tensor(t.m_tensor) {
        t.m_handle = nullptr;
        t.m_tensor = dlpack::tensor();
    }

    tensor &operator=(tensor &&t) noexcept {
        detail::tensor_dec_ref(m_handle);
        m_handle = t.m_handle;
        m_tensor = t.m_tensor;
        t.m_handle = nullptr;
        t.m_tensor = dlpack::tensor();
        return *this;
    }

    tensor &operator=(const tensor &t) {
        detail::tensor_inc_ref(t.m_handle);
        detail::tensor_dec_ref(m_handle);
        m_handle = t.m_handle;
        m_tensor = t.m_tensor;
    }

    dlpack::dtype dtype() const { return m_tensor.dtype; }
    size_t ndim() const { return m_tensor.ndim; }
    size_t shape(size_t i) const { return m_tensor.shape[i]; }
    int64_t stride(size_t i) const { return m_tensor.strides[i]; }
    bool is_valid() const { return m_handle != nullptr; }
    int32_t device_type() const { return m_tensor.device.device_type; }
    int32_t device_id() const { return m_tensor.device.device_id; }
    detail::tensor_handle *handle() const { return m_handle; }

    const void *data() const {
        return (const uint8_t *) m_tensor.data + m_tensor.byte_offset;
    }
    void *data() { return (uint8_t *) m_tensor.data + m_tensor.byte_offset; }

    template <typename... Ts>
    auto& operator()(Ts... indices) {
        static_assert(
            !std::is_same<typename Info::scalar_type, void>::value,
            "To use nb::tensor::operator(), you must add a scalar type "
            "annotation (e.g. 'float') to the tensor template parameters.");
        static_assert(
            !std::is_same<typename Info::shape_type, void>::value,
            "To use nb::tensor::operator(), you must add a nb::shape<> "
            "annotation to the tensor template parameters.");
        static_assert(sizeof...(Ts) == Info::shape_type::size,
                      "nb::tensor::operator(): invalid number of arguments");

        int64_t counter = 0, index = 0;
        ((index += int64_t(indices) * m_tensor.strides[counter++]), ...);
        return (typename Info::scalar_type &) *(
            (uint8_t *) m_tensor.data + m_tensor.byte_offset +
            index * sizeof(typename Info::scalar_type));
    }

private:
    detail::tensor_handle *m_handle = nullptr;
    dlpack::tensor m_tensor;
};


PYBIND11_NAMESPACE_BEGIN(detail)

template<typename... Args>
struct tensor_caster<tensor<Args...>> {
    using type = tensor<Args...>;

    type value;

    static constexpr auto name = const_name("dl_managed_tensor");

    template <typename T>
    using cast_op_type = type;

    explicit operator type&() { return value; }
    
    bool load(handle src, bool convert) {
        constexpr size_t size = (0 + ... + detail::tensor_arg<Args>::size);
        size_t shape[size + 1];
        detail::tensor_req req;
        req.shape = shape;
        (detail::tensor_arg<Args>::apply(req), ...);
        value = tensor<Args...>(tensor_import(src.ptr(), req, convert));
        return value.is_valid();
        
    }

    static handle cast(type& src, return_value_policy /* policy */, handle /* parent */) {
        return tensor_wrap(src.handle(), int(Value::Info::framework));
    }
}

// template <>
// struct type_caster<DLManagedTensor> {
//  public:
//  protected:
//   DLManagedTensor* value;

//  public:
//   static constexpr auto name = const_name("dl_managed_tensor");

//   template <typename T>
//   using cast_op_type = DLManagedTensor*&;

//   explicit operator DLManagedTensor* &() { return value; }

//   bool load(handle src, bool)
//   {
//     pybind11::capsule capsule;
//     if (pybind11::isinstance<pybind11::capsule>(src)) {
//       capsule = pybind11::reinterpret_borrow<pybind11::capsule>(src);
//     } else if (pybind11::hasattr(src, "__dlpack__")) {
//       // note that, if the user tries to pass in an object with
//       // a __dlpack__ attribute instead of a capsule, they have
//       // no ability to pass the "stream" argument to __dlpack__

//       // this can cause a performance reduction, but not a
//       // correctness error, since the default null stream will
//       // be used for synchronization if no stream is passed

//       // https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.__dlpack__.html

//       // I think I'm stealing this. The result of CallMethod
//       // should already have a reference count of 1
//       capsule = pybind11::reinterpret_steal<pybind11::capsule>(
//           PyObject_CallMethod(src.ptr(), "__dlpack__", nullptr));
//     } else {
//       std::cerr << "pybind11_dlpack_caster.h: not a capsule or a __dlpack__ object" << std::endl;
//       return false;
//     }

//     // is this the same as PyCapsule_IsValid?
//     if (strcmp(capsule.name(), "dltensor") != 0) {
//       return false;
//     }
//     value = capsule.get_pointer<DLManagedTensor>();
//     capsule.set_name("used_dltensor");
//     return true;
//   }

//   static handle cast(DLManagedTensor* src, return_value_policy /* policy */, handle /* parent */)
//   {
//     if (src) {
//     // why call release here?
//     // need to get the capsule a name!
//       pybind11::capsule capsule(src, reinterpret_cast<void (*)(void*)>(src->deleter));
//       capsule.set_name("dltensor");
//       return capsule.release();

//     } else {
//       return pybind11::none().inc_ref();
//     }
//   }
// };

// template <>
// struct type_caster<tensor> {
//  public:
//  protected:
//   DLManagedTensor* value;

//  public:
//   static constexpr auto name = const_name("dl_managed_tensor");

//   template <typename T>
//   using cast_op_type = DLManagedTensor*&;

//   explicit operator DLManagedTensor* &() { return value; }

//   bool load(handle src, bool)
//   {
//     pybind11::capsule capsule;
//     if (pybind11::isinstance<pybind11::capsule>(src)) {
//       capsule = pybind11::reinterpret_borrow<pybind11::capsule>(src);
//     } else if (pybind11::hasattr(src, "__dlpack__")) {
//       // note that, if the user tries to pass in an object with
//       // a __dlpack__ attribute instead of a capsule, they have
//       // no ability to pass the "stream" argument to __dlpack__

//       // this can cause a performance reduction, but not a
//       // correctness error, since the default null stream will
//       // be used for synchronization if no stream is passed

//       // https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.__dlpack__.html

//       // I think I'm stealing this. The result of CallMethod
//       // should already have a reference count of 1
//       capsule = pybind11::reinterpret_steal<pybind11::capsule>(
//           PyObject_CallMethod(src.ptr(), "__dlpack__", nullptr));
//     } else {
//       std::cerr << "pybind11_dlpack_caster.h: not a capsule or a __dlpack__ object" << std::endl;
//       return false;
//     }

//     // is this the same as PyCapsule_IsValid?
//     if (strcmp(capsule.name(), "dltensor") != 0) {
//       return false;
//     }
//     value = capsule.get_pointer<DLManagedTensor>();
//     capsule.set_name("used_dltensor");
//     return true;
//   }

//   static handle cast(DLManagedTensor* src, return_value_policy /* policy */, handle /* parent */)
//   {
//     if (src) {
//     // why call release here?
//     // need to get the capsule a name!
//       pybind11::capsule capsule(src, reinterpret_cast<void (*)(void*)>(src->deleter));
//       capsule.set_name("dltensor");
//       return capsule.release();

//     } else {
//       return pybind11::none().inc_ref();
//     }
//   }
// };

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
