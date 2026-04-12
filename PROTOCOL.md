# FLUX Cooperative Intelligence Protocol (FCIP)

## The DIVIDE-CONQUER-SYNTHESIZE (DCS) Protocol

A structured protocol for multi-agent collective problem-solving. DCS enables
agents with different specializations to decompose complex problems, work on
sub-problems in parallel, and synthesize partial results into coherent solutions.

### Design Philosophy

1. **Emergence over orchestration**: The protocol provides structure, not
   micromanagement. Agents self-select into roles based on capability matching.
2. **Graceful degradation**: If an agent drops out or fails, its sub-problem is
   reclaimed without collapsing the entire effort.
3. **Provenance by default**: Every partial result carries its methodology,
   assumptions, and confidence — synthesis is never blind aggregation.
4. **Learning loop**: Each cooperative session feeds back into trust scores and
   decomposition heuristics, improving future collaborations.

---

### Phase 1: PROBLEM DECOMPOSITION

**Actor**: ProblemOwner (the agent that initiates the protocol)

1. The ProblemOwner receives or identifies a complex problem.
2. Analyzes the problem structure and decomposes it into sub-problems.
3. Creates a **ProblemManifest** containing:
   - `problem_id`: Unique identifier
   - `statement`: Natural-language problem description
   - `owner`: Agent ID of the ProblemOwner
   - `sub_problems`: List of SubProblem descriptors
   - `required_capabilities`: Union of all capabilities needed
   - `difficulty`: Overall difficulty estimate (0.0–1.0)
   - `created_at`: Timestamp
4. Broadcasts the ProblemManifest via the BROADCAST opcode.

**SubProblem decomposition heuristics**:
- Each sub-problem should be independently solvable by a single agent.
- Dependencies between sub-problems are explicitly modeled as a DAG.
- Sub-problems should have roughly equal effort to maximize parallelism.
- Sub-problems at the leaves of the dependency graph should require the fewest
  capabilities each (specialization), while integration sub-problems may require
  broader knowledge.

---

### Phase 2: AGENT SELF-SELECTION

**Actor**: All available agents in the fleet

1. Each agent evaluates the ProblemManifest against its own capabilities.
2. Agents CLAIM sub-problems they can solve by sending a **Claim message** to
   the ProblemOwner containing:
   - `sub_problem_id`: The sub-problem being claimed
   - `agent_id`: Claiming agent
   - `estimated_effort`: Time/effort estimate
   - `confidence`: 0.0–1.0 self-assessed probability of solving correctly
   - `approach`: Brief description of planned method
   - `evidence`: Prior results or credentials supporting the claim
3. The ProblemOwner collects all claims and resolves conflicts:
   - **Conflict**: Two agents claim the same sub-problem.
   - **Resolution priority**:
     1. Higher confidence + evidence score
     2. Lower estimated effort (efficiency)
     3. Random tiebreaker
4. Unclaimed sub-problems may be:
   - Left pending (if non-critical)
   - Force-assigned to the most capable available agent
   - Decomposed further into smaller sub-problems
5. ProblemOwner TELLS each agent their assignment.

**Assignment optimization**: The ProblemOwner may use a matching algorithm
(weighted bipartite matching) to maximize overall confidence across assignments,
rather than greedily assigning the most-confident agent to each sub-problem.

---

### Phase 3: PARALLEL EXECUTION

**Actor**: Assigned agents (workers)

1. Each assigned agent works on their sub-problem independently.
2. Agents **can** communicate during execution:
   - **ASK**: Request help from another agent (e.g., clarification, shared data).
   - **TELL**: Send progress updates to the ProblemOwner.
   - **TELL**: Share intermediate results that may help other agents.
3. If an agent encounters an insurmountable obstacle, it may **YIELD** its
   sub-problem back to the ProblemOwner with a reason. The ProblemOwner then
   re-enters Phase 2 for that sub-problem only.
4. Agents working on dependent sub-problems wait for their dependencies to
   complete (or request early partial results via ASK).
5. Each agent produces a **PartialResult**:
   - `sub_problem_id`: Which sub-problem was solved
   - `agent_id`: Solving agent
   - `answer`: The partial answer
   - `confidence`: 0.0–1.0
   - `methodology`: How the answer was derived
   - `assumptions`: What was assumed (explicitly stated)
   - `elapsed_time`: Time spent

---

### Phase 4: RESULT COLLECTION

**Actor**: ProblemOwner

1. ProblemOwner collects all PartialResults from assigned agents.
2. Checks for missing results (timed-out or crashed agents).
3. For missing results, ProblemOwner may:
   - Retry with a different agent (loop to Phase 2 for that sub-problem)
   - Proceed with incomplete information (lower overall confidence)
4. Assembles all partial results into a **ResultCollection**:
   - Complete set of PartialResults
   - Coverage map (which sub-problems were solved)
   - Gaps identified
   - Overall completeness score

---

### Phase 5: SYNTHESIS

**Actor**: ProblemOwner or designated Synthesizer

1. Combines all partial results into a coherent final answer.
2. **Conflict detection**: Identifies contradictions between agent answers.
3. **Conflict resolution strategies**:
   - **Weighted vote**: Combine answers weighted by agent confidence and trust.
   - **Evidence arbitration**: The answer with stronger supporting evidence wins.
   - **Decompose further**: If conflict is fundamental, decompose the conflicting
     aspect into a new sub-problem and loop to Phase 2.
   - **Synthesis fusion**: Where answers are complementary rather than
     contradictory, merge them into a richer combined answer.
4. Produces a **SynthesizedAnswer**:
   - `answer`: The combined final answer
   - `confidence`: Overall confidence (product of sub-confidence minimums or
     weighted average, depending on problem type)
   - `provenance`: Map from each part of the answer to the agent(s) responsible
   - `conflicts_resolved`: How any disagreements were handled
   - `methodology`: Narrative of how the answer was assembled

---

### Phase 6: VERIFICATION

**Actor**: Verification agent or automated test harness

1. If available, a verification agent (or a different agent than those who
   contributed) checks the final answer.
2. Verification strategies:
   - **Test case execution**: Run known-good test cases against the answer.
   - **Independent solution**: Another agent solves the same problem from
     scratch and results are compared.
   - **Consistency check**: Verify the answer is internally consistent and
     doesn't violate known constraints.
   - **Peer review**: Another agent critiques the methodology and assumptions.
3. Produces a **VerificationResult**:
   - `passed`: Boolean
   - `score`: 0.0–1.0 quality score
   - `issues`: List of specific problems found
   - `suggestions`: Recommendations for improvement
4. If verification fails:
   - Problems are fed back as refined sub-problems.
   - Protocol loops back to Phase 2 (not Phase 1 — the decomposition may be
     partially reusable).

---

### Phase 7: LEARNING

**Actor**: System / ProblemOwner

1. Records the full session in a **SessionLog**:
   - Problem, decomposition, assignments, partial results, synthesis, verification
   - Total time, total agents involved
   - Which strategies worked and which didn't
2. **Trust score updates**:
   - Agents whose partial results survived synthesis and verification get
     increased trust.
   - Agents whose results were contradicted or failed verification get
     decreased trust.
   - Trust scores influence future claim priority (Phase 2).
3. **Decomposition pattern learning**:
   - If a particular decomposition pattern led to high verification scores,
     store it as a template for similar future problems.
   - Build a library of proven decomposition strategies.
4. **Capability refinement**:
   - Update agent capability profiles based on observed performance.
   - An agent that consistently solves math sub-problems gains a stronger
     "math" capability signal.

---

## Alternative Collaboration Patterns

### MAP-REDUCE Pattern

For **embarrassingly parallel** problems where the same operation applies to
many independent data items.

```
ProblemOwner.map(data_items, operation) -> distributed work
Workers.apply(operation, item) -> partial results
ProblemOwner.reduce(partial_results, reduction_fn) -> final answer
```

**When to use**:
- Batch processing of independent items
- Each item requires the same analysis
- No dependencies between items

**Example**: Analyze 100 code files for security vulnerabilities.

---

### DEBATE Pattern

For problems where agents **disagree** and need to argue their positions before
converging.

```
Step 1: PROPOSE — Each relevant agent proposes their answer with supporting evidence
Step 2: ARGUE   — Agents present arguments for/against each proposal (structured rounds)
Step 3: VOTE    — Agents vote on best proposal (weighted by trust/confidence)
Step 4: RESOLVE — Majority wins; dissenting opinions are recorded in provenance
```

**Debate rules**:
- Each argument round is time-bounded to prevent infinite debate.
- Agents must cite evidence — bare assertions carry zero weight.
- An agent may change their vote if persuaded by stronger evidence.
- The final resolution includes a "dissent register" — opinions that lost but
  were well-argued are preserved for future reference.

**When to use**:
- Design decisions with trade-offs
- Diagnosing ambiguous issues
- Any problem where reasonable agents might legitimately disagree

**Termination**: Debate ends when:
- All agents agree (consensus), or
- Maximum rounds reached, or
- Confidence convergence — vote distribution stabilizes across rounds.

---

### CASCADE Pattern

For **sequential** processing where each agent builds on the previous agent's
output.

```
Agent A processes input -> intermediate result 1
Agent B takes result 1, adds value -> intermediate result 2
Agent C takes result 2, finalizes -> final answer
```

**Feedback mechanism**: Any agent in the chain can request a revision from an
earlier agent by sending a YIELD-BACK message with specific feedback.

**When to use**:
- Pipeline processing (e.g., outline -> draft -> edit -> proofread)
- Problems with natural sequential dependencies
- When each stage requires different expertise

---

## Message Format Reference

All cooperative messages extend the base FLUX message format:

```
{
  "opcode": "CLAIM | YIELD | PARTIAL_RESULT | VERIFICATION | ...",
  "session_id": "uuid of the cooperative session",
  "problem_id": "uuid of the problem",
  "payload": { ... type-specific fields ... }
}
```

### New Opcodes

| Opcode        | Direction              | Purpose                              |
|---------------|------------------------|--------------------------------------|
| `MANIFEST`    | ProblemOwner → All     | Broadcast problem decomposition      |
| `CLAIM`       | Worker → ProblemOwner  | Agent claims a sub-problem           |
| `ASSIGN`      | ProblemOwner → Worker  | Confirm/deny claim                   |
| `YIELD`       | Worker → ProblemOwner  | Give up a sub-problem                |
| `PROGRESS`    | Worker → ProblemOwner  | Status update                        |
| `PARTIAL`     | Worker → ProblemOwner  | Submit partial result                |
| `SYNTHESIZE`  | ProblemOwner → Synth.  | Request synthesis                    |
| `VERIFY`      | Owner → Verifier       | Request verification                 |
| `DEBATE_PROP` | Any → All              | Propose a debate position            |
| `DEBATE_ARG`  | Any → All              | Argue for/against a proposal         |
| `DEBATE_VOTE` | Any → Tally            | Cast a vote                          |

---

## Confidence Arithmetic

Confidence propagation through the protocol follows these rules:

1. **Parallel combination** (independent sub-problems):
   `C_total = min(C_1, C_2, ..., C_n)` — chain is only as strong as its
   weakest link.

2. **Weighted combination** (synthesis with trust):
   `C_synthesis = sum(C_i * T_i) / sum(T_i)` — where T_i is the trust score
   of agent i.

3. **Sequential cascade**:
   `C_cascade = product(C_1, C_2, ..., C_n)` — errors compound in pipelines.

4. **Debate consensus**:
   `C_debate = fraction_of_agents_agreeing * avg_confidence_of_agreeing_agents`

---

## Trust Score System

Trust is per-agent, per-capability, and evolves over time:

```
trust(agent, capability) = base_trust
  + sum(contribution_scores) / max(total_contributions, 1)
  - penalty_factor * failure_count
```

Where:
- `base_trust`: Default 0.5 (neutral)
- `contribution_scores`: Each successful partial result adds to this
- `failure_count`: Each verification failure or contradiction increments this
- `penalty_factor`: Decay weight (e.g., 0.1 — failures matter but don't dominate)
- Trust is bounded to [0.0, 1.0]

Trust scores influence:
1. Claim priority in Phase 2 (higher trust = preferential assignment)
2. Synthesis weighting in Phase 5 (higher trust = more influence on final answer)
3. Debate vote weighting (higher trust = heavier vote)
4. Agent selection for verification (lower trust agents verify higher trust)

---

## Session Lifecycle

```
[ProblemOwner receives problem]
        |
        v
  Phase 1: Decompose
        |
        v
  Phase 2: Self-Select
        |
        v
  Phase 3: Execute (parallel)
        |    ^
        |    | (retry on YIELD)
        |    |
        v    |
  Phase 4: Collect
        |
        v
  Phase 5: Synthesize
        |
        v
  Phase 6: Verify ──fail──> Phase 2 (refined)
        |
       pass
        |
        v
  Phase 7: Learn
        |
        v
  [Solution delivered]
```

---

## Error Handling

| Situation                | Response                                      |
|--------------------------|-----------------------------------------------|
| Agent crashes mid-task   | YIELD timeout triggers reassignment           |
| No agent claims sub-prob | Force-assign or decompose further             |
| All agents fail sub-prob | Report to ProblemOwner, mark as unresolvable  |
| Synthesis contradicts    | Trigger debate or further decomposition       |
| Verification loop (3x)   | Accept best-effort with reduced confidence    |
| Circular dependencies    | Detected in Phase 1, decompose to break cycle |
| Timeout (session-wide)   | Return partial results with completeness score |
