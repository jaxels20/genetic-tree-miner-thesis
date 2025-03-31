#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "PetriNet.hpp"  // Include your PetriNet classes
#include "Eventlog.hpp"  // Include your EventLog classes
#include "token_based_replay.cpp"  // Include your token_based_replay function
#include "precision.cpp"  // Include your precision function
namespace py = pybind11;

PYBIND11_MODULE(FastTokenBasedReplay, m) {
    py::class_<Place>(m, "Place")
        .def(py::init<std::string, int>(), py::arg("name"), py::arg("tokens") = 0)
        .def("add_tokens", &Place::add_tokens)
        .def("remove_tokens", &Place::remove_tokens)
        .def("__repr__", &Place::repr);

    py::class_<Transition>(m, "Transition")
        .def(py::init<std::string>(), py::arg("name") = "")
        .def("__repr__", &Transition::repr);

    py::class_<Arc>(m, "Arc")
        .def(py::init<std::string, std::string, int>(), py::arg("source"), py::arg("target"), py::arg("weight") = 1)
        .def("__repr__", &Arc::repr);

    py::class_<PetriNet>(m, "PetriNet")
        .def(py::init<>())
        .def("add_place", &PetriNet::add_place)
        .def("add_transition", &PetriNet::add_transition)
        .def("add_arc", &PetriNet::add_arc)
        .def("__repr__", &PetriNet::repr)
        .def("set_initial_marking", &PetriNet::set_initial_marking)
        .def("set_final_marking", &PetriNet::set_final_marking);

    py::class_<Event>(m, "Event")
        .def(py::init<std::string, std::string, std::unordered_map<std::string, std::string>>())
        .def("__repr__", &Event::repr);

    py::class_<Trace>(m, "Trace")
        .def(py::init<std::string, std::unordered_map<std::string, std::string>>())
        .def("add_event", &Trace::add_event)
        .def("__repr__", &Trace::repr);

    py::class_<EventLog>(m, "EventLog")
        .def(py::init<>())
        .def("add_trace", &EventLog::add_trace)
        .def("__repr__", &EventLog::repr);
    
    py::class_<Marking>(m, "Marking")
        .def(py::init<>())
        .def(py::init<std::initializer_list<std::pair<std::string, uint32_t>>>())
        .def("add_place", &Marking::add_place)
        .def("number_of_tokens", &Marking::number_of_tokens);

    m.def("calculate_fitness", &calculate_fitness);
    m.def("calculate_precision", &calculate_precision);
}
