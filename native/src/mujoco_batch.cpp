#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace multiverse {

class DynamicLibrary {
 public:
  explicit DynamicLibrary(const std::string& path) {
#ifdef _WIN32
    handle_ = LoadLibraryA(path.c_str());
    if (!handle_) {
      throw std::runtime_error("Unable to load MuJoCo runtime: " + path);
    }
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      const char* error = dlerror();
      throw std::runtime_error("Unable to load MuJoCo runtime " + path + ": " +
                               (error ? error : "unknown loader error"));
    }
#endif
  }

  ~DynamicLibrary() {
    if (!handle_) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  template <typename Function>
  Function symbol(const char* name) const {
#ifdef _WIN32
    auto address = GetProcAddress(static_cast<HMODULE>(handle_), name);
#else
    auto address = dlsym(handle_, name);
#endif
    if (!address) throw std::runtime_error(std::string("MuJoCo is missing symbol ") + name);
    return reinterpret_cast<Function>(address);
  }

 private:
#ifdef _WIN32
  HMODULE handle_ = nullptr;
#else
  void* handle_ = nullptr;
#endif
};

struct MujocoApi {
  explicit MujocoApi(const std::string& library_path) : library(library_path) {
    load_model = library.symbol<decltype(load_model)>("mj_loadXML");
    make_data = library.symbol<decltype(make_data)>("mj_makeData");
    delete_data = library.symbol<decltype(delete_data)>("mj_deleteData");
    delete_model = library.symbol<decltype(delete_model)>("mj_deleteModel");
    reset_data = library.symbol<decltype(reset_data)>("mj_resetData");
    forward = library.symbol<decltype(forward)>("mj_forward");
    step = library.symbol<decltype(step)>("mj_step");
  }

  DynamicLibrary library;
  decltype(&mj_loadXML) load_model = nullptr;
  decltype(&mj_makeData) make_data = nullptr;
  decltype(&mj_deleteData) delete_data = nullptr;
  decltype(&mj_deleteModel) delete_model = nullptr;
  decltype(&mj_resetData) reset_data = nullptr;
  decltype(&mj_forward) forward = nullptr;
  decltype(&mj_step) step = nullptr;
};

class MujocoBatch {
 public:
  MujocoBatch(const std::string& model_path, const std::string& library_path,
              std::size_t env_count, int worker_count)
      : api_(std::make_unique<MujocoApi>(library_path)), env_count_(env_count) {
    if (env_count_ == 0) throw std::invalid_argument("env_count must be positive");
    char error[2048] = {};
    model_ = api_->load_model(model_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) {
      throw std::runtime_error("MuJoCo could not load model '" + model_path + "': " + error);
    }
    try {
      data_.reserve(env_count_);
      for (std::size_t lane = 0; lane < env_count_; ++lane) {
        mjData* item = api_->make_data(model_);
        if (!item) throw std::runtime_error("MuJoCo failed to allocate mjData");
        data_.push_back(item);
      }
      const auto requested = worker_count > 0
                                 ? static_cast<std::size_t>(worker_count)
                                 : static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
      worker_count_ = std::min(env_count_, requested);
      for (std::size_t index = 0; index < worker_count_; ++index) {
        workers_.emplace_back([this, index] { worker_loop(index); });
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~MujocoBatch() { cleanup(); }
  MujocoBatch(const MujocoBatch&) = delete;
  MujocoBatch& operator=(const MujocoBatch&) = delete;

  std::size_t env_count() const { return env_count_; }
  std::size_t worker_count() const { return worker_count_; }
  int nq() const { return model_->nq; }
  int nv() const { return model_->nv; }
  int nu() const { return model_->nu; }

  py::tuple step(const py::array_t<double, py::array::c_style | py::array::forcecast>& controls,
                 int substeps) {
    if (controls.ndim() != 2 || controls.shape(0) != static_cast<py::ssize_t>(env_count_) ||
        controls.shape(1) != model_->nu) {
      throw py::value_error("controls must have shape (env_count, model.nu)");
    }
    if (substeps < 1) throw py::value_error("substeps must be positive");
    const auto input = controls.unchecked<2>();
    std::vector<mjtNum> copied(env_count_ * static_cast<std::size_t>(model_->nu));
    for (std::size_t lane = 0; lane < env_count_; ++lane) {
      for (int actuator = 0; actuator < model_->nu; ++actuator) {
        copied[lane * static_cast<std::size_t>(model_->nu) + actuator] = input(lane, actuator);
      }
    }

    {
      py::gil_scoped_release release;
      run_parallel([this, &copied, substeps](std::size_t worker) {
        const std::size_t begin = env_count_ * worker / worker_count_;
        const std::size_t end = env_count_ * (worker + 1) / worker_count_;
        for (std::size_t lane = begin; lane < end; ++lane) {
          mjData* item = data_[lane];
          for (int actuator = 0; actuator < model_->nu; ++actuator) {
            item->ctrl[actuator] = copied[lane * static_cast<std::size_t>(model_->nu) + actuator];
          }
          for (int iteration = 0; iteration < substeps; ++iteration) api_->step(model_, item);
        }
      });
    }
    return state_arrays();
  }

  py::tuple reset(const std::vector<std::size_t>& lanes) {
    for (std::size_t lane : lanes) {
      if (lane >= env_count_) throw py::index_error("lane index is out of range");
    }
    {
      py::gil_scoped_release release;
      run_parallel([this, &lanes](std::size_t worker) {
        for (std::size_t i = worker; i < lanes.size(); i += worker_count_) {
          api_->reset_data(model_, data_[lanes[i]]);
          api_->forward(model_, data_[lanes[i]]);
        }
      });
    }
    return state_arrays();
  }

  py::tuple set_state(const py::array_t<double, py::array::c_style | py::array::forcecast>& qpos,
                      const py::array_t<double, py::array::c_style | py::array::forcecast>& qvel) {
    if (qpos.ndim() != 2 || qvel.ndim() != 2 ||
        qpos.shape(0) != static_cast<py::ssize_t>(env_count_) ||
        qvel.shape(0) != static_cast<py::ssize_t>(env_count_) || qpos.shape(1) != model_->nq ||
        qvel.shape(1) != model_->nv) {
      throw py::value_error("qpos/qvel must have shape (env_count, model.nq/nv)");
    }
    const auto positions = qpos.unchecked<2>();
    const auto velocities = qvel.unchecked<2>();
    {
      py::gil_scoped_release release;
      run_parallel([this, &positions, &velocities](std::size_t worker) {
        const std::size_t begin = env_count_ * worker / worker_count_;
        const std::size_t end = env_count_ * (worker + 1) / worker_count_;
        for (std::size_t lane = begin; lane < end; ++lane) {
          for (int i = 0; i < model_->nq; ++i) data_[lane]->qpos[i] = positions(lane, i);
          for (int i = 0; i < model_->nv; ++i) data_[lane]->qvel[i] = velocities(lane, i);
          api_->forward(model_, data_[lane]);
        }
      });
    }
    return state_arrays();
  }

  py::tuple state() const { return state_arrays(); }

 private:
  void run_parallel(std::function<void(std::size_t)> task) {
    if (workers_.size() == 1 && env_count_ == 1) {
      task(0);
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    task_ = std::move(task);
    remaining_ = workers_.size();
    ++generation_;
    task_ready_.notify_all();
    task_done_.wait(lock, [this] { return remaining_ == 0; });
    task_ = {};
  }

  void worker_loop(std::size_t index) {
    std::size_t seen_generation = 0;
    while (true) {
      std::function<void(std::size_t)> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        task_ready_.wait(lock, [this, &seen_generation] {
          return stopping_ || generation_ != seen_generation;
        });
        if (stopping_) return;
        seen_generation = generation_;
        task = task_;
      }
      task(index);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--remaining_ == 0) task_done_.notify_one();
      }
    }
  }

  py::tuple state_arrays() const {
    py::array_t<double> qpos({static_cast<py::ssize_t>(env_count_),
                              static_cast<py::ssize_t>(model_->nq)});
    py::array_t<double> qvel({static_cast<py::ssize_t>(env_count_),
                              static_cast<py::ssize_t>(model_->nv)});
    auto positions = qpos.mutable_unchecked<2>();
    auto velocities = qvel.mutable_unchecked<2>();
    for (std::size_t lane = 0; lane < env_count_; ++lane) {
      for (int i = 0; i < model_->nq; ++i) positions(lane, i) = data_[lane]->qpos[i];
      for (int i = 0; i < model_->nv; ++i) velocities(lane, i) = data_[lane]->qvel[i];
    }
    return py::make_tuple(std::move(qpos), std::move(qvel));
  }

  void cleanup() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      task_ready_.notify_all();
    }
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
    if (api_) {
      for (mjData* item : data_) api_->delete_data(item);
      data_.clear();
      if (model_) api_->delete_model(model_);
    }
    model_ = nullptr;
  }

  std::unique_ptr<MujocoApi> api_;
  mjModel* model_ = nullptr;
  std::vector<mjData*> data_;
  std::size_t env_count_ = 0;
  std::size_t worker_count_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable task_ready_;
  std::condition_variable task_done_;
  std::function<void(std::size_t)> task_;
  std::size_t generation_ = 0;
  std::size_t remaining_ = 0;
  bool stopping_ = false;
};

}  // namespace multiverse

PYBIND11_MODULE(_multiverse_native, module) {
  module.doc() = "Multiverse's optional C++17 MuJoCo batch runtime";
  py::class_<multiverse::MujocoBatch>(module, "MujocoBatch")
      .def(py::init<const std::string&, const std::string&, std::size_t, int>(),
           py::arg("model_path"), py::arg("library_path"), py::arg("env_count"),
           py::arg("worker_count") = 0)
      .def_property_readonly("env_count", &multiverse::MujocoBatch::env_count)
      .def_property_readonly("worker_count", &multiverse::MujocoBatch::worker_count)
      .def_property_readonly("nq", &multiverse::MujocoBatch::nq)
      .def_property_readonly("nv", &multiverse::MujocoBatch::nv)
      .def_property_readonly("nu", &multiverse::MujocoBatch::nu)
      .def("step", &multiverse::MujocoBatch::step, py::arg("controls"), py::arg("substeps") = 1)
      .def("reset", &multiverse::MujocoBatch::reset, py::arg("lanes"))
      .def("set_state", &multiverse::MujocoBatch::set_state, py::arg("qpos"), py::arg("qvel"))
      .def("state", &multiverse::MujocoBatch::state);
}
