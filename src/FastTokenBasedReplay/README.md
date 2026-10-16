# FastTokenBasedReplay

Token-based replay fitness and precision for Petri nets, implemented in C++ with Python bindings.
It was built as the fitness engine of the Genetic Tree Miner (GTM), where it evaluates each
candidate model 18–70× faster than pm4py's pure-Python token replay and ETC precision.

## Installation

From this directory (needs a C++17 compiler; CMake and pybind11 are fetched automatically):

```bash
pip install .
```

## Usage with pm4py

```python
import pm4py
import FastTokenBasedReplay as ftr

df = pm4py.read_xes("log.xes")
net, im, fm = pm4py.discover_petri_net_inductive(df)

log = ftr.from_pm4py_log(df)              # pm4py EventLog or DataFrame
ftr_net = ftr.from_pm4py_net(net, im, fm)

fitness = ftr.calculate_fitness(log, ftr_net)
precision = ftr.calculate_precision(log, ftr_net)
```

`from_pm4py_log` takes the usual `activity_key`, `case_id_key` and `timestamp_key` arguments.
pm4py itself is not a dependency: the converters only use the attributes of pm4py's objects.

## Usage without pm4py

```python
import FastTokenBasedReplay as ftr

net = ftr.PetriNet()
for p in ["start", "p1", "end"]:
    net.add_place(ftr.Place(p, 0))
for t in ["A", "B"]:
    net.add_transition(ftr.Transition(t))
for source, target in [("start", "A"), ("A", "p1"), ("p1", "B"), ("B", "end")]:
    net.add_arc(ftr.Arc(source, target, 1))

initial, final = ftr.Marking(), ftr.Marking()
initial.add_place("start", 1)
final.add_place("end", 1)
net.set_initial_marking(initial)
net.set_final_marking(final)

log = ftr.from_traces([["A", "B"], ["B", "A"]])
ftr.calculate_fitness(log, net)
```

Transitions are identified by name. A visible transition is named after its activity, and every
transition whose name starts with `tau` is silent. `from_pm4py_net` handles this naming and
rejects nets it cannot represent (duplicate visible labels, or visible labels starting with `tau`).

`calculate_fitness` also accepts `prefix_caching=True` or `suffix_caching=True` (not both).

## Relation to pm4py

The results match pm4py's token-based replay fitness and ETC precision on many nets, but not on
all of them: on the Sepsis log, for example, some GTM models get different scores. The cause has
not been investigated yet.

## Development

```bash
pip install -e ".[test]" && pytest tests/python      # Python tests

cmake -S . -B build -DFTR_BUILD_TESTS=ON -DFTR_BUILD_PYTHON=OFF
cmake --build build && ctest --test-dir build      # C++ tests (GoogleTest is downloaded if missing)
```

Wheels for Linux, macOS and Windows are built with cibuildwheel by the
`fast-token-replay-wheels` GitHub Actions workflow.
