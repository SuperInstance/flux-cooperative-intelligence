# FLUX Cooperative Intelligence Protocol (FCIP)

> Novel multi-agent cooperative problem-solving — collective intelligence
> through structured collaboration.

## Overview

FCIP is a protocol for **collective problem-solving** in AI agent fleets. While
existing fleet primitives (ASK, TELL, BROADCAST) handle point-to-point
communication, FCIP provides the **coordination layer** that enables multiple
agents to work together on complex problems, share partial results, and converge
on solutions.

## The Core Protocol: DIVIDE-CONQUER-SYNTHESIZE (DCS)

The DCS protocol orchestrates cooperative problem-solving in seven phases:

```
Problem → Decompose → Self-Select → Execute → Collect → Synthesize → Verify → Learn → Solution
```

| Phase | What Happens | Key Op |
|-------|-------------|--------|
| **1. Decompose** | ProblemOwner breaks problem into sub-problems | `MANIFEST` |
| **2. Self-Select** | Agents claim sub-problems based on capabilities | `CLAIM` / `ASSIGN` |
| **3. Execute** | Agents solve sub-problems in parallel | `PROGRESS` / `PARTIAL` |
| **4. Collect** | ProblemOwner gathers all partial results | `PARTIAL` |
| **5. Synthesize** | Combine results, resolve conflicts | trust-weighted merge |
| **6. Verify** | Independent agent checks the answer | `VERIFY` |
| **7. Learn** | Update trust scores, store patterns | internal |

### Key Design Decisions

- **Agent self-selection** rather than central assignment — agents know their
  own strengths best.
- **Conflict resolution via trust-weighted arbitration** — agents with
  proven track records get more influence.
- **Graceful degradation** — if an agent crashes, its sub-problem is
  reclaimed without collapsing the session.
- **Provenance by default** — every partial result carries its methodology,
  assumptions, and confidence score.
- **Verification loop** — solutions are independently checked; failures
  trigger targeted retries rather than full restarts.

See [PROTOCOL.md](./PROTOCOL.md) for the full specification.

## Alternative Patterns

Not every problem needs full DCS. Three alternative patterns handle common
special cases:

### MapReduce Pattern
For **embarrassingly parallel** problems — the same operation applied to many
independent items.
```
items → [Agent.map(item) for each item] → reduce(results)
```
**Example**: Analyze 100 files for security vulnerabilities.

### Debate Pattern
For problems where agents **disagree** and need to argue positions.
```
Propose → Argue → Vote → Resolve (majority wins, dissent recorded)
```
**Example**: Choosing between React vs Vue for a frontend.

### Cascade Pattern
For **sequential pipelines** where each agent builds on the previous.
```
Agent A → Agent B → Agent C (with optional feedback loops)
```
**Example**: Outline → Draft → Edit → Proofread.

### Pattern Selection
The `PatternSelector` analyzes a problem statement and recommends the best
pattern:

```python
from protocol.patterns import PatternSelector

selector = PatternSelector()
rec = selector.recommend("Analyze each of the 100 files independently")
# → PatternRecommendation(pattern_name="map_reduce", confidence=0.75, ...)
```

## Quick Start

### Problem Decomposition

```python
from protocol.problem import ProblemDecomposer

decomposer = ProblemDecomposer()

# Decompose a complex problem
manifest = decomposer.decompose(
    "Design the architecture, implement the backend, "
    "and write documentation for a REST API"
)

print(f"Sub-problems: {len(manifest.sub_problems)}")
print(f"Capabilities needed: {manifest.required_capabilities}")
print(f"Difficulty: {manifest.difficulty:.2f}")
```

### Full DCS Execution

```python
from protocol.executor import DCSExecutor

# Agents must implement: agent_id, capabilities, evaluate_claim(), solve(), verify()
executor = DCSExecutor(agents=[math_agent, code_agent, verifier_agent])

solution = executor.run("Calculate the optimal algorithm and implement it")

print(f"Answer: {solution.answer}")
print(f"Confidence: {solution.confidence:.2f}")
print(f"Contributors: {solution.agent_contributions}")
print(f"Time: {solution.total_time:.2f}s")
```

### MapReduce Pattern

```python
from protocol.patterns import MapReducePattern

pattern = MapReducePattern(agents=[agent1, agent2, agent3])
result = pattern.run(
    data_items=[1, 2, 3, 4, 5],
    map_fn=lambda agent, item: item * agent.multiplier,
    reduce_fn=lambda results: sum(results),
)
print(f"Result: {result.reduced_answer}")
```

### Debate Pattern

```python
from protocol.patterns import DebatePattern, DebateProposal

pattern = DebatePattern(
    agents=[agent_a, agent_b, agent_c],
    trust_weights={"agent_a": 1.5, "agent_b": 1.0, "agent_c": 0.8},
)
result = pattern.run(
    problem_statement="Choose the best sorting algorithm",
    propose_fn=lambda agent, problem: DebateProposal(
        agent_id=agent.agent_id,
        answer=agent.preferred_algorithm,
        evidence=agent.evidence,
        confidence=agent.confidence,
    ),
)
print(f"Winner: {result.winning_answer}")
print(f"Dissent: {result.dissent_register}")
```

### Cascade Pattern

```python
from protocol.patterns import CascadePattern

pipeline = CascadePattern(agent_chain=[outliner, drafter, editor, proofreader])
result = pipeline.run(
    initial_input="Write a blog post about AI",
    process_fn=lambda agent, data: agent.process(data),
)
print(f"Final: {result.final_answer}")
print(f"Feedback loops: {result.feedback_loops}")
```

## Architecture

```
src/
  protocol/
    __init__.py          # Package exports
    problem.py           # Data types + ProblemDecomposer
    executor.py          # DCS protocol executor + TrustManager + Synthesizer
    patterns.py          # MapReduce, Debate, Cascade patterns + PatternSelector
  tests/
    test_cooperative_intelligence.py  # Comprehensive test suite
```

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `ProblemManifest` | `problem` | Full problem specification with sub-problems |
| `SubProblem` | `problem` | Individual sub-problem with status tracking |
| `PartialResult` | `problem` | Agent's output for a sub-problem |
| `ProblemDecomposer` | `problem` | Decomposes problems into sub-problems |
| `DCSExecutor` | `executor` | Runs the full 7-phase DCS protocol |
| `Synthesizer` | `executor` | Combines partial results with conflict resolution |
| `TrustManager` | `executor` | Per-agent, per-capability trust scores |
| `MapReducePattern` | `patterns` | Embarrassingly parallel pattern |
| `DebatePattern` | `patterns` | Structured debate with voting |
| `CascadePattern` | `patterns` | Sequential pipeline with feedback |
| `PatternSelector` | `patterns` | Recommends the best pattern for a problem |

## Trust Score System

Trust is per-agent, per-capability, and evolves over time:

```
trust(agent, capability) = base_trust + reward_history - penalty_history
```

Trust influences:
- **Claim priority** in Phase 2
- **Synthesis weighting** in Phase 5
- **Debate vote weight** in Debate pattern
- **Verifier selection** in Phase 6

## Running Tests

```bash
cd src && python -m pytest tests/ -v
# or
cd src/tests && python test_cooperative_intelligence.py -v
```

## Design Principles

1. **Emergence over orchestration** — Structure, not micromanagement.
2. **Graceful degradation** — Partial results are better than no results.
3. **Provenance by default** — Every answer comes with its reasoning.
4. **Continuous learning** — Every session improves future sessions.
5. **No external dependencies** — Pure Python, framework-agnostic.

## Future Directions

- **LLM-powered decomposer**: Replace heuristic decomposition with
  LLM-generated sub-problems.
- **Fleet integration**: Connect to real FLUX ASK/TELL/BROADCAST opcodes.
- **Persistent trust store**: Save trust scores across sessions.
- **Adaptive pattern selection**: Learn which patterns work best for which
  problem types.
- **Time-boxed phases**: Enforce SLAs on each protocol phase.
