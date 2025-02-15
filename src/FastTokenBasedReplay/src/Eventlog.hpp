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
};

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
};

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
