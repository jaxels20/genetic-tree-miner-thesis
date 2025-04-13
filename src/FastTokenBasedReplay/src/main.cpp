#include "token_based_replay.cpp"
#include "precision.cpp"
#include "PetriNet.hpp"
#include "Eventlog.hpp"

int main() {

    PetriNet net;
    net.add_place(Place("start", 0));
    net.add_place(Place("p1", 0));
    net.add_place(Place("end", 0));

    net.add_transition(Transition("A"));
    net.add_transition(Transition("B"));
    net.add_transition(Transition("tau1"));


    net.add_arc(Arc("start", "tau1"));
    net.add_arc(Arc("tau1", "p1"));
    net.add_arc(Arc("p1", "A"));
    net.add_arc(Arc("A", "p1"));
    net.add_arc(Arc("p1", "B"));
    net.add_arc(Arc("B", "end"));


    net.set_initial_marking(Marking({{"start", 1}}));
    net.set_final_marking(Marking({{"end", 1}}));

    std::string repeaded_A_100 = std::string(5000, 'A');
    std::string repeaded_A_500 = std::string(5001, 'A');
    std::string repeaded_A_1000 = std::string(5002, 'A');
    

    std::vector<std::string> trace_list = {repeaded_A_100 + "B", repeaded_A_500 + "B", repeaded_A_1000 + "B",};
    EventLog eventlog = EventLog::from_trace_list(trace_list);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double, std::micro> elapsed;
    double fitness = 0.0;

    // time the function
    start = std::chrono::high_resolution_clock::now();
    fitness = calculate_fitness(eventlog, net, false, false);
    end = std::chrono::high_resolution_clock::now();
    // elapsed in ms
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time (no caching): " << elapsed.count() << " Micro" << std::endl;

    std::cout << "Fitness: " << fitness << std::endl;

    // time the function
    start = std::chrono::high_resolution_clock::now();
    fitness = calculate_fitness(eventlog, net, true, false);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time (prefix): " << elapsed.count() << " Micro" << std::endl;
    std::cout << "Fitness: " << fitness << std::endl;

    // // time the function
    // start = std::chrono::high_resolution_clock::now();
    // fitness = calculate_fitness(eventlog, net, false, true);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Elapsed time (suffix): " << elapsed.count() << " Micro" << std::endl;
    // std::cout << "Fitness: " << fitness << std::endl;


    // // time the function
    // start = std::chrono::high_resolution_clock::now();
    // fitness = calculate_fitness(eventlog, net, true, true);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Elapsed time (prefix+suffix): " << elapsed.count() << " Micro" << std::endl;
    // std::cout << "Fitness: " << fitness << std::endl;

    return 0;
}