#pragma once

#include <vector>
#include <unordered_map>

class Event {
public:
    std::string activity;
    std::string timestamp;
    std::unordered_map<std::string, std::string> attributes;

    Event(std::string activity, std::string timestamp,
          std::unordered_map<std::string, std::string> attributes)
        : activity(activity), timestamp(timestamp), attributes(attributes) {}

    std::string repr() const {
        return "Event(activity=" + activity + ", timestamp=" + timestamp + ")";
    }

    // Overloading the == operator for comparing events
    bool operator==(const Event& other) const {
        return activity == other.activity;
    }

};

// Hash function for Event 
namespace std {
    template<>
    struct hash<Event> {
        std::size_t operator()(const Event& e) const {
            return std::hash<std::string>()(e.activity);
        }
    };
}

class Trace {
public:
    std::string trace_id;
    std::vector<Event> events;
    std::unordered_map<std::string, std::string> attributes;

    Trace(std::string trace_id, std::unordered_map<std::string, std::string> attributes)
        : trace_id(trace_id), attributes(attributes) {}

    void add_event(const Event& event) {
        events.push_back(event);
    }
   
    std::string repr() const {
        return "Trace(trace_id=" + trace_id + ", events=" + std::to_string(events.size()) + ")";
    }

    // Overloading the == operator for comparing traces
    bool operator==(const Trace& other) const {
        return events == other.events;
    }

};

// Hash function for Trace (only considering events)
namespace std {
    template<>
    struct hash<Trace> {
        std::size_t operator()(const Trace& t) const {
            std::size_t seed = 0;
            for (const auto& event : t.events) {
                seed ^= std::hash<Event>()(event) + 0x9e3779b9 + (seed << 6) + (seed >> 2); 
            }
            return seed;
        }
    };
}

class EventLog {
public:
    std::vector<Trace> traces;

    void add_trace(const Trace& trace) {
        traces.push_back(trace);
    }

    std::string repr() const {
        return "EventLog(traces=" + std::to_string(traces.size()) + ")";
    }

    static EventLog from_trace_list(const std::vector<std::string>& trace_list) {
        EventLog eventlog;
        
        for (size_t i = 0; i < trace_list.size(); ++i) {
            std::string trace_id = "trace_" + std::to_string(i + 1);
            Trace trace(trace_id, {});
            
            for (char activity : trace_list[i]) {
                Event event(std::string(1, activity), "", {});
                trace.add_event(event);
            }
            
            eventlog.add_trace(trace);
        }
        
        return eventlog;
    }
};
