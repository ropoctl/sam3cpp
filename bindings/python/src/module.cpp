#include "sam3/pipeline.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

class PyPrediction {
public:
    explicit PyPrediction(sam3::Sam3Prediction pred) : pred_(std::move(pred)) {}

    int width() const { return pred_.width; }
    int height() const { return pred_.height; }
    int count() const { return pred_.count; }

    py::array_t<float> scores() const {
        py::array_t<float> out({static_cast<py::ssize_t>(pred_.scores.size())});
        if (!pred_.scores.empty()) {
            std::memcpy(out.mutable_data(), pred_.scores.data(), pred_.scores.size() * sizeof(float));
        }
        return out;
    }

    py::array_t<float> boxes_xyxy() const {
        py::array_t<float> out({static_cast<py::ssize_t>(pred_.count), static_cast<py::ssize_t>(4)});
        if (!pred_.boxes_xyxy.empty()) {
            std::memcpy(out.mutable_data(), pred_.boxes_xyxy.data(), pred_.boxes_xyxy.size() * sizeof(float));
        }
        return out;
    }

    py::array_t<float> masks_chw() const {
        py::array_t<float> out({
            static_cast<py::ssize_t>(pred_.count),
            static_cast<py::ssize_t>(pred_.height),
            static_cast<py::ssize_t>(pred_.width),
        });
        if (!pred_.masks.empty()) {
            std::memcpy(out.mutable_data(), pred_.masks.data(), pred_.masks.size() * sizeof(float));
        }
        return out;
    }

    py::array_t<float> masks_hwc() const {
        py::array_t<float> out({
            static_cast<py::ssize_t>(pred_.height),
            static_cast<py::ssize_t>(pred_.width),
            static_cast<py::ssize_t>(pred_.count),
        });

        const size_t mask_size = static_cast<size_t>(pred_.width) * static_cast<size_t>(pred_.height);
        float * dst = out.mutable_data();
        for (int c = 0; c < pred_.count; ++c) {
            const float * src = pred_.masks.data() + static_cast<size_t>(c) * mask_size;
            for (int y = 0; y < pred_.height; ++y) {
                for (int x = 0; x < pred_.width; ++x) {
                    const size_t src_idx = static_cast<size_t>(y) * static_cast<size_t>(pred_.width) + static_cast<size_t>(x);
                    const size_t dst_idx = (src_idx * static_cast<size_t>(pred_.count)) + static_cast<size_t>(c);
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
        return out;
    }

    py::array_t<float> mask(int index) const {
        if (index < 0 || index >= pred_.count) {
            throw py::index_error("mask index out of range");
        }

        const size_t mask_size = static_cast<size_t>(pred_.width) * static_cast<size_t>(pred_.height);
        py::array_t<float> out({static_cast<py::ssize_t>(pred_.height), static_cast<py::ssize_t>(pred_.width)});
        if (mask_size > 0) {
            const float * src = pred_.masks.data() + static_cast<size_t>(index) * mask_size;
            std::memcpy(out.mutable_data(), src, mask_size * sizeof(float));
        }
        return out;
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Prediction(count=" << pred_.count
            << ", width=" << pred_.width
            << ", height=" << pred_.height << ")";
        return oss.str();
    }

private:
    sam3::Sam3Prediction pred_;
};

class PySam3Model {
public:
    PySam3Model(const std::string & gguf_path, bool prefer_gpu = true, const std::string & bpe_path = {})
        : pipeline_(std::make_unique<sam3::Sam3ImagePipeline>(gguf_path, prefer_gpu, bpe_path)) {}

    PyPrediction predict(const std::string & image_path, const std::string & prompt) const {
        return PyPrediction(pipeline_->predict(image_path, prompt));
    }

    PyPrediction predict_tokens(const std::string & image_path, const std::vector<int32_t> & tokens) const {
        return PyPrediction(pipeline_->predict_tokens(image_path, tokens));
    }

    PyPrediction predict_points(const std::string & image_path,
                                py::array_t<float> points_xy,
                                py::array_t<int32_t> labels) const {
        auto pts = points_xy.unchecked<2>();
        auto lbl = labels.unchecked<1>();
        if (pts.shape(0) != lbl.shape(0) || pts.shape(1) != 2) {
            throw std::runtime_error("points_xy must be (N, 2) and labels must be (N,)");
        }
        const int32_t count = static_cast<int32_t>(pts.shape(0));
        return PyPrediction(pipeline_->predict_points(image_path, pts.data(0, 0), lbl.data(0), count));
    }

    PyPrediction predict_box(const std::string & image_path,
                             float x1, float y1, float x2, float y2) const {
        return PyPrediction(pipeline_->predict_box(image_path, x1, y1, x2, y2));
    }

private:
    std::unique_ptr<sam3::Sam3ImagePipeline> pipeline_;
};

}  // namespace

PYBIND11_MODULE(_sam3cpp, m) {
    m.doc() = "pybind11 bindings for sam3cpp";

    py::class_<PyPrediction>(m, "Prediction")
        .def_property_readonly("width", &PyPrediction::width)
        .def_property_readonly("height", &PyPrediction::height)
        .def_property_readonly("count", &PyPrediction::count)
        .def_property_readonly("scores", &PyPrediction::scores)
        .def_property_readonly("boxes_xyxy", &PyPrediction::boxes_xyxy)
        .def_property_readonly("masks", &PyPrediction::masks_hwc)
        .def_property_readonly("masks_chw", &PyPrediction::masks_chw)
        .def("mask", &PyPrediction::mask, py::arg("index"))
        .def("__len__", &PyPrediction::count)
        .def("__repr__", &PyPrediction::repr);

    py::class_<PySam3Model>(m, "Sam3Model")
        .def(py::init<const std::string &, bool, const std::string &>(),
             py::arg("gguf_path"),
             py::arg("prefer_gpu") = true,
             py::arg("bpe_path") = std::string())
        .def("predict", &PySam3Model::predict, py::arg("image_path"), py::arg("prompt"))
        .def("predict_tokens", &PySam3Model::predict_tokens, py::arg("image_path"), py::arg("tokens"))
        .def("predict_points", &PySam3Model::predict_points, py::arg("image_path"), py::arg("points_xy"), py::arg("labels"),
             "Point-prompted segmentation. points_xy is (N,2) normalized [0,1], labels is (N,) with 1=positive, 0=negative.")
        .def("predict_box", &PySam3Model::predict_box, py::arg("image_path"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             "Box-prompted segmentation. Coordinates in pixels (xyxy format).");
}
