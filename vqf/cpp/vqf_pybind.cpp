//
// Created by wayne on 2023/6/7.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "vqf.hpp"


#ifndef VQF_SINGLE_PRECISION
typedef double vqf_real_t;
#else
typedef float vqf_real_t;
#endif

namespace py = pybind11;

PYBIND11_MODULE(vqf_py, m) {
    py::class_<VQF>(m, "VQF")
            .def(py::init<vqf_real_t, vqf_real_t, vqf_real_t>(),
                 py::arg("gyrTs"), py::arg("accTs") = -1.0, py::arg("magTs") = -1.0)
            .def(py::init<const VQFParams &, vqf_real_t, vqf_real_t, vqf_real_t>(),
                 py::arg("params"), py::arg("gyrTs"), py::arg("accTs") = -1.0, py::arg("magTs") = -1.0)
            .def("updateGyr", [](VQF &self, const vqf_real_t dt, py::array_t<vqf_real_t> gyr) {
                py::buffer_info info = gyr.request();
                if (info.ndim != 1)
                    throw std::runtime_error("gyr must be a 1-dimensional array");
                self.updateGyr(dt, static_cast<vqf_real_t *>(info.ptr));
            })
            .def("updateAcc", [](VQF &self, const vqf_real_t dt, py::array_t<vqf_real_t> acc) {
                py::buffer_info info = acc.request();
                if (info.ndim != 1)
                    throw std::runtime_error("acc must be a 1-dimensional array");
                self.updateAcc(dt, static_cast<vqf_real_t *>(info.ptr));
            })
            .def("updateMag", [](VQF &self, const vqf_real_t dt, py::array_t<vqf_real_t> mag) {
                py::buffer_info info = mag.request();
                if (info.ndim != 1)
                    throw std::runtime_error("mag must be a 1-dimensional array");
                self.updateMag(dt, static_cast<vqf_real_t *>(info.ptr));
            })
            .def("update", [](VQF &self, const vqf_real_t dt, py::array_t<vqf_real_t> gyr, py::array_t<vqf_real_t> acc) {
                py::buffer_info gyrInfo = gyr.request();
                py::buffer_info accInfo = acc.request();
                if (gyrInfo.ndim != 1 || accInfo.ndim != 1)
                    throw std::runtime_error("gyr and acc must be 1-dimensional arrays");
                self.update(dt, static_cast<vqf_real_t *>(gyrInfo.ptr), static_cast<vqf_real_t *>(accInfo.ptr));
            })
            .def("update", [](VQF &self, const vqf_real_t dt, py::array_t<vqf_real_t> gyr, py::array_t<vqf_real_t> acc, py::array_t<vqf_real_t> mag) {
                py::buffer_info gyrInfo = gyr.request();
                py::buffer_info accInfo = acc.request();
                py::buffer_info magInfo = mag.request();
                if (gyrInfo.ndim != 1 || accInfo.ndim != 1 || magInfo.ndim != 1)
                    throw std::runtime_error("gyr, acc, and mag must be 1-dimensional arrays");
                self.update(dt, static_cast<vqf_real_t *>(gyrInfo.ptr), static_cast<vqf_real_t *>(accInfo.ptr), static_cast<vqf_real_t *>(magInfo.ptr));
            })
            .def("getQuat3D", [](const VQF &self) {
                vqf_real_t out[4];
                self.getQuat3D(out);
                return py::array_t<vqf_real_t>({4}, out);
            })
            .def("getQuat6D", [](const VQF &self) {
                vqf_real_t out[4];
                self.getQuat6D(out);
                return py::array_t<vqf_real_t>({4}, out);
            })
            .def("getQuat9D", [](const VQF &self) {
                vqf_real_t out[4];
                self.getQuat9D(out);
                return py::array_t<vqf_real_t>({4}, out);
            })
            .def("getDelta", &VQF::getDelta)
            .def("getBiasEstimate", [](const VQF &self) {
                vqf_real_t out[3];
                self.getBiasEstimate(out);
                return py::array_t<vqf_real_t>({3}, out);
            })
            .def("setBiasEstimate", [](VQF &self, py::array_t<vqf_real_t> bias, vqf_real_t sigma) {
                py::buffer_info info = bias.request();
                if (info.ndim != 1 || info.size != 3)
                    throw std::runtime_error("bias must be a 1-dimensional array with size 3");
                self.setBiasEstimate(static_cast<vqf_real_t *>(info.ptr), sigma);
            })
            .def("getRestDetected", &VQF::getRestDetected)
            .def("getMagDistDetected", &VQF::getMagDistDetected)
            .def("getRelativeRestDeviations", [](const VQF &self) {
                vqf_real_t out[2];
                self.getRelativeRestDeviations(out);
                return py::array_t<vqf_real_t>({2}, out);
            })
            .def("getMagRefNorm", &VQF::getMagRefNorm)
            .def("getMagRefDip", &VQF::getMagRefDip)
            .def("setMagRef", &VQF::setMagRef)
            .def("setTauAcc", &VQF::setTauAcc)
            .def("setTauMag", &VQF::setTauMag)
            .def("setRestBiasEstEnabled", &VQF::setRestBiasEstEnabled)
            .def("setMagDistRejectionEnabled", &VQF::setMagDistRejectionEnabled)
            .def("setRestDetectionThresholds", &VQF::setRestDetectionThresholds)
            .def("getDebugVars", &VQF::getDebugVars)
            .def("getParams", &VQF::getParams)
            .def("getCoeffs", &VQF::getCoeffs)
            .def("getState", &VQF::getState)
            .def("setState", &VQF::setState)
            .def("resetState", &VQF::resetState)
            .def_static("quatMultiply", &VQF::quatMultiply)
            .def_static("quatConj", &VQF::quatConj)
            .def_static("quatSetToIdentity", &VQF::quatSetToIdentity)
            .def_static("quatApplyDelta", &VQF::quatApplyDelta)
            .def_static("quatRotate", &VQF::quatRotate)
            .def_static("norm", &VQF::norm)
            .def_static("normalize", &VQF::normalize)
            .def_static("clip", &VQF::clip)
            .def_static("gainFromTau", &VQF::gainFromTau)
            .def_static("filterCoeffs", &VQF::filterCoeffs)
            .def_static("filterInitialState", &VQF::filterInitialState)
            .def_static("filterAdaptStateForCoeffChange", &VQF::filterAdaptStateForCoeffChange)
            .def_static("filterStep", &VQF::filterStep)
            .def_static("filterVec", &VQF::filterVec);

    py::class_<VQFParams>(m, "VQFParams")
            .def(py::init<>())
            .def_readwrite("tauAcc", &VQFParams::tauAcc)
            .def_readwrite("tauMag", &VQFParams::tauMag)
            .def_readwrite("motionBiasEstEnabled", &VQFParams::motionBiasEstEnabled)
            .def_readwrite("restBiasEstEnabled", &VQFParams::restBiasEstEnabled)
            .def_readwrite("magDistRejectionEnabled", &VQFParams::magDistRejectionEnabled)
            .def_readwrite("biasForgettingTime", &VQFParams::biasForgettingTime)
            .def_readwrite("biasClip", &VQFParams::biasClip)
            .def_readwrite("biasSigmaMotion", &VQFParams::biasSigmaMotion)
            .def_readwrite("biasVerticalForgettingFactor", &VQFParams::biasVerticalForgettingFactor)
            .def_readwrite("biasSigmaRest", &VQFParams::biasSigmaRest)
            .def_readwrite("restMinT", &VQFParams::restMinT)
            .def_readwrite("restFilterTau", &VQFParams::restFilterTau)
            .def_readwrite("restThGyr", &VQFParams::restThGyr)
            .def_readwrite("restThAcc", &VQFParams::restThAcc)
            .def_readwrite("magCurrentTau", &VQFParams::magCurrentTau)
            .def_readwrite("magRefTau", &VQFParams::magRefTau)
            .def_readwrite("magNormTh", &VQFParams::magNormTh)
            .def_readwrite("magDipTh", &VQFParams::magDipTh)
            .def_readwrite("magNewTime", &VQFParams::magNewTime)
            .def_readwrite("magNewFirstTime", &VQFParams::magNewFirstTime)
            .def_readwrite("magNewMinGyr", &VQFParams::magNewMinGyr)
            .def_readwrite("magMinUndisturbedTime", &VQFParams::magMinUndisturbedTime)
            .def_readwrite("magMaxRejectionTime", &VQFParams::magMaxRejectionTime)
            .def_readwrite("magRejectionFactor", &VQFParams::magRejectionFactor);

    py::class_<VQFState>(m, "VQFState")
            .def_property_readonly("gyrQuat", [](VQFState &self) {
                return py::array_t<vqf_real_t>({4}, self.gyrQuat);
            })
            .def_property_readonly("accQuat", [](VQFState &self) {
                return py::array_t<vqf_real_t>({4}, self.accQuat);
            })
            .def_readwrite("delta", &VQFState::delta)
            .def_readwrite("restDetected", &VQFState::restDetected)
            .def_readwrite("magDistDetected", &VQFState::magDistDetected)
            .def_property_readonly("lastAccLp", [](VQFState &self) {
                return py::array_t<vqf_real_t>({3}, self.lastAccLp);
            })
            .def_property_readonly("accLpState", [](VQFState &self) {
                return py::array_t<double>({3, 2}, self.accLpState);
            })
            .def_readwrite("lastAccCorrAngularRate", &VQFState::lastAccCorrAngularRate)
            .def_readwrite("kMagInit", &VQFState::kMagInit)
            .def_readwrite("lastMagDisAngle", &VQFState::lastMagDisAngle)
            .def_readwrite("lastMagCorrAngularRate", &VQFState::lastMagCorrAngularRate)
            .def_property_readonly("bias", [](VQFState &self) {
                return py::array_t<vqf_real_t>({3}, self.bias);
            })
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
            .def_property_readonly("biasP", [](VQFState &self) {
                return py::array_t<vqf_real_t>({9}, self.biasP);
            })
#else
                    .def_readwrite("biasP", &VQFState::biasP)
#endif
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
            .def_property_readonly("motionBiasEstRLpState", [](VQFState &self) {
                return py::array_t<double>({9, 2}, self.motionBiasEstRLpState);
            })
            .def_property_readonly("motionBiasEstBiasLpState", [](VQFState &self) {
                return py::array_t<double>({2, 2}, self.motionBiasEstBiasLpState);
            })
#endif
            .def_property_readonly("restLastSquaredDeviations", [](VQFState &self) {
                return py::array_t<vqf_real_t>({2}, self.restLastSquaredDeviations);
            })
            .def_readwrite("restT", &VQFState::restT)
            .def_property_readonly("restLastGyrLp", [](VQFState &self) {
                return py::array_t<vqf_real_t>({3}, self.restLastGyrLp);
            })
            .def_property_readonly("restGyrLpState", [](VQFState &self) {
                return py::array_t<double>({3, 2}, self.restGyrLpState);
            })
            .def_property_readonly("restLastAccLp", [](VQFState &self) {
                return py::array_t<vqf_real_t>({3}, self.restLastAccLp);
            })
            .def_property_readonly("restAccLpState", [](VQFState &self) {
                return py::array_t<double>({3, 2}, self.restAccLpState);
            })
            .def_readwrite("magRefNorm", &VQFState::magRefNorm)
            .def_readwrite("magRefDip", &VQFState::magRefDip)
            .def_readwrite("magUndisturbedT", &VQFState::magUndisturbedT)
            .def_readwrite("magRejectT", &VQFState::magRejectT)
            .def_readwrite("magCandidateNorm", &VQFState::magCandidateNorm)
            .def_readwrite("magCandidateDip", &VQFState::magCandidateDip)
            .def_readwrite("magCandidateT", &VQFState::magCandidateT)
            .def_property_readonly("magNormDip", [](VQFState &self) {
                return py::array_t<vqf_real_t>({2}, self.magNormDip);
            })
            .def_property_readonly("magNormDipLpState", [](VQFState &self) {
                return py::array_t<double>({2, 2}, self.magNormDipLpState);
            });

    py::class_<VQFCoefficients>(m, "VQFCoefficients")
            .def_readwrite("gyrTs", &VQFCoefficients::gyrTs)
            .def_readwrite("accTs", &VQFCoefficients::accTs)
            .def_readwrite("magTs", &VQFCoefficients::magTs)
            .def_property_readonly("accLpB", [](VQFCoefficients &self) {
                return py::array_t<double>({3}, self.accLpB);
            })
            .def_property_readonly("accLpA", [](VQFCoefficients &self) {
                return py::array_t<double>({2}, self.accLpA);
            })
            .def_readwrite("kMag", &VQFCoefficients::kMag)
            .def_readwrite("biasP0", &VQFCoefficients::biasP0)
            .def_readwrite("biasV", &VQFCoefficients::biasV)
            .def_property_readonly("restGyrLpB", [](VQFCoefficients &self) {
                return py::array_t<double>({3}, self.restGyrLpB);
            })
            .def_property_readonly("restGyrLpA", [](VQFCoefficients &self) {
                return py::array_t<double>({2}, self.restGyrLpA);
            })
            .def_property_readonly("restAccLpB", [](VQFCoefficients &self) {
                return py::array_t<double>({3}, self.restAccLpB);
            })
            .def_property_readonly("restAccLpA", [](VQFCoefficients &self) {
                return py::array_t<double>({2}, self.restAccLpA);
            })
            .def_readwrite("kMagRef", &VQFCoefficients::kMagRef);

    py::class_<VQFDebug>(m, "VQFDebug")
            .def_readwrite("k", &VQFDebug::k)
            .def_readwrite("magYaw", &VQFDebug::magYaw);
}
