#pragma once

#include "ext/nanobind/include/nanobind/nanobind.h"
#include "ext/nanobind/include/nanobind/eigen/dense.h"
#include "lfda.h"

namespace nb = nanobind;


NB_MODULE(lfda, m) {
    nb::class_<LFDA>(m, "LFDA")
        .def(nb::init<int, int, EMBEDDING_TYPE>(),
            nb::arg("n_components"), nb::arg("k"), nb::arg("embedding"),
            "Constructor for the LFDA class")
        .def("fit", &LFDA::Fit,
            nb::arg("Xin"), nb::arg("yin"),
            "Method to fit the LFDA model.")
        .def("transform", &LFDA::Transform,
            nb::arg("X"),
            "Method to transform a query input.");

    nb::enum_<EMBEDDING_TYPE>(m, "EMBEDDING_TYPE")
        .value("WEIGHTED", EMBEDDING_TYPE::WEIGHTED)
        .value("PLAIN", EMBEDDING_TYPE::PLAIN)
        .value("ORTHONORMALIZED", EMBEDDING_TYPE::ORTHONORMALIZED);
}
