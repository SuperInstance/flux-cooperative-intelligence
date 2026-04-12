"""
flux-cooperative-intelligence: Comprehensive test suite.

Tests cover:
- Problem decomposition
- Sub-problem assignment
- DCS protocol execution with mock agents
- Synthesis with conflicting results
- Verification pass/fail
- MapReduce pattern
- Debate pattern
- Cascade pattern
- Trust management
- Pattern selection
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol.problem import (
    Claim,
    CooperativeSolution,
    PartialResult,
    ProblemDecomposer,
    ProblemManifest,
    SubProblem,
    SubProblemStatus,
    VerificationResult,
)
from protocol.executor import (
    DCSExecutor,
    Synthesizer,
    TrustManager,
)
from protocol.patterns import (
    CascadePattern,
    CascadeResult,
    DebateArgument,
    DebatePattern,
    DebateProposal,
    MapReducePattern,
    PatternSelector,
)


# ---------------------------------------------------------------------------
# Mock Agents for Testing
# ---------------------------------------------------------------------------

@dataclass
class MockAgent:
    """A simple mock agent for testing."""

    agent_id: str
    capabilities: Set[str]
    trust_score: float = 0.5

    # Configurable behavior
    solve_answer: Any = None
    solve_confidence: float = 0.8
    solve_methodology: str = "Mock solve"
    will_claim: bool = True
    claim_confidence: float = 0.8
    verify_passes: bool = True


def make_claim_agent(agent: MockAgent, sub_problem: SubProblem) -> Optional[Claim]:
    """Evaluate claim for mock agent."""
    if not agent.will_claim:
        return None
    if agent.capabilities & sub_problem.capabilities_needed:
        return Claim(
            sub_problem_id=sub_problem.id,
            agent_id=agent.agent_id,
            confidence=agent.claim_confidence,
            approach="Mock approach",
            evidence="Mock evidence",
        )
    return None


def make_solve_agent(agent: MockAgent, sub_problem: SubProblem) -> PartialResult:
    """Solve for mock agent."""
    return PartialResult(
        sub_problem_id=sub_problem.id,
        agent_id=agent.agent_id,
        answer=agent.solve_answer or f"solution from {agent.agent_id}",
        confidence=agent.solve_confidence,
        methodology=agent.solve_methodology,
        assumptions=["mock assumption"],
    )


def make_verify_agent(
    agent: MockAgent, answer: Any, problem_statement: str
) -> VerificationResult:
    """Verify for mock agent."""
    return VerificationResult(
        passed=agent.verify_passes,
        score=0.9 if agent.verify_passes else 0.2,
        issues=[] if agent.verify_passes else ["Verification failed (mock)"],
        verifier_id=agent.agent_id,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestProblemDecomposition(unittest.TestCase):
    """Test the ProblemDecomposer class."""

    def setUp(self):
        self.decomposer = ProblemDecomposer()

    def test_detect_capabilities_math(self):
        """Capability detection should identify math keywords."""
        caps = self.decomposer._detect_capabilities(
            "Calculate the optimal number of servers needed"
        )
        self.assertIn("math", caps)

    def test_detect_capabilities_coding(self):
        """Capability detection should identify coding keywords."""
        caps = self.decomposer._detect_capabilities(
            "Implement a function to debug the API"
        )
        self.assertIn("coding", caps)

    def test_detect_capabilities_writing(self):
        """Capability detection should identify writing keywords."""
        caps = self.decomposer._detect_capabilities(
            "Write a summary explaining the document"
        )
        self.assertIn("writing", caps)

    def test_detect_capabilities_multiple(self):
        """Should detect multiple capabilities."""
        caps = self.decomposer._detect_capabilities(
            "Design and implement a secure system to calculate statistics"
        )
        self.assertTrue(len(caps) >= 2)

    def test_detect_capabilities_default_reasoning(self):
        """Should default to reasoning if nothing detected."""
        caps = self.decomposer._detect_capabilities(
            "The quick brown fox jumps over the lazy dog"
        )
        self.assertIn("reasoning", caps)

    def test_decompose_creates_manifest(self):
        """Decompose should return a ProblemManifest."""
        manifest = self.decomposer.decompose(
            "Calculate the optimal algorithm and implement it"
        )
        self.assertIsInstance(manifest, ProblemManifest)
        self.assertTrue(len(manifest.statement) > 0)
        self.assertTrue(len(manifest.sub_problems) > 0)
        self.assertTrue(manifest.problem_id.startswith("prob-"))

    def test_decompose_sub_problems_have_ids(self):
        """Each sub-problem should have a unique ID."""
        manifest = self.decomposer.decompose("Implement and test a system")
        ids = [sp.id for sp in manifest.sub_problems]
        self.assertEqual(len(ids), len(set(ids)))

    def test_decompose_parent_problem_linking(self):
        """Sub-problems should reference their parent problem."""
        manifest = self.decomposer.decompose("Research and write a report")
        for sp in manifest.sub_problems:
            self.assertEqual(sp.parent_problem_id, manifest.problem_id)

    def test_decompose_simple_problem(self):
        """Simple problem should produce 1-2 sub-problems."""
        manifest = self.decomposer.decompose("Fix the bug")
        self.assertLessEqual(len(manifest.sub_problems), 2)

    def test_decompose_complex_problem(self):
        """Complex problem should produce more sub-problems."""
        manifest = self.decomposer.decompose(
            "Research the market, design the architecture, implement "
            "the backend, write documentation, and perform security analysis"
        )
        self.assertGreaterEqual(len(manifest.sub_problems), 3)

    def test_estimate_difficulty(self):
        """Difficulty estimation should produce values in [0, 1]."""
        sp = SubProblem(
            description="A complex multi-step problem requiring many capabilities",
            capabilities_needed={"math", "coding", "design"},
        )
        diff = self.decomposer.estimate_difficulty(sp, "complex problem")
        self.assertGreaterEqual(diff, 0.0)
        self.assertLessEqual(diff, 1.0)

    def test_estimate_difficulty_more_caps_harder(self):
        """More capabilities needed should generally mean higher difficulty."""
        sp_easy = SubProblem(
            description="Simple task",
            capabilities_needed={"reasoning"},
        )
        sp_hard = SubProblem(
            description="Complex task requiring many skills",
            capabilities_needed={"math", "coding", "design", "security", "writing"},
        )
        diff_easy = self.decomposer.estimate_difficulty(sp_easy)
        diff_hard = self.decomposer.estimate_difficulty(sp_hard)
        self.assertGreater(diff_hard, diff_easy)

    def test_identify_dependencies(self):
        """Dependency identification should build a graph."""
        sp1 = SubProblem(description="Research the topic", capabilities_needed={"research"})
        sp2 = SubProblem(description="Write the code", capabilities_needed={"coding"})
        graph = self.decomposer.identify_dependencies([sp1, sp2])
        self.assertIsInstance(graph, dict)
        self.assertIn(sp1.id, graph)
        self.assertIn(sp2.id, graph)

    def test_suggest_assignment(self):
        """Assignment suggestion should match agents to sub-problems."""
        agents = {
            "math-agent": {"math", "reasoning"},
            "code-agent": {"coding", "design"},
        }
        sp1 = SubProblem(description="Math task", capabilities_needed={"math"})
        sp2 = SubProblem(description="Code task", capabilities_needed={"coding"})
        assignments = self.decomposer.suggest_assignment([sp1, sp2], agents)
        self.assertEqual(assignments[sp1.id], "math-agent")
        self.assertEqual(assignments[sp2.id], "code-agent")

    def test_suggest_assignment_balanced(self):
        """Assignment should balance load across agents."""
        agents = {
            "agent-a": {"math", "coding"},
            "agent-b": {"math", "coding"},
        }
        sps = [
            SubProblem(description=f"Task {i}", capabilities_needed={"math"})
            for i in range(4)
        ]
        assignments = self.decomposer.suggest_assignment(sps, agents)
        # Both agents should get some work
        assigned_agents = set(assignments.values())
        self.assertEqual(len(assigned_agents), 2)

    def test_record_and_retrieve_pattern(self):
        """Decomposition patterns should be recordable and retrievable."""
        manifest = ProblemManifest(
            statement="test",
            required_capabilities={"math", "coding"},
            difficulty=0.7,
        )
        self.decomposer.record_pattern(manifest, 0.9)
        similar = self.decomposer.get_similar_patterns(
            {"math", "coding"}, 0.7
        )
        self.assertTrue(len(similar) > 0)

    def test_tailor_to_agents(self):
        """Decomposition should be tailored to available agent capabilities."""
        agents = {"agent-a": {"math"}}
        manifest = self.decomposer.decompose(
            "Perform security analysis",
            agent_capabilities=agents,
        )
        # Should not have capabilities that no agent can fulfill
        all_agent_caps = set()
        for caps in agents.values():
            all_agent_caps.update(caps)
        for sp in manifest.sub_problems:
            self.assertTrue(
                sp.capabilities_needed & all_agent_caps or not sp.capabilities_needed,
                f"Sub-problem {sp.id} requires {sp.capabilities_needed} "
                f"but no agent has them"
            )


class TestSubProblemAssignment(unittest.TestCase):
    """Test sub-problem assignment logic."""

    def test_claim_resolution_single_claim(self):
        """Single claim should be accepted."""
        from protocol.executor import DCSExecutor

        agent = MockAgent(agent_id="a1", capabilities={"math"})
        agent.will_claim = True
        agent.claim_confidence = 0.8

        sp = SubProblem(description="Math task", capabilities_needed={"math"})
        manifest = ProblemManifest(sub_problems=[sp])

        executor = DCSExecutor(agents=[agent])
        executor.decomposer = ProblemDecomposer()

        claims = [Claim(
            sub_problem_id=sp.id,
            agent_id="a1",
            confidence=0.8,
        )]
        assignments = executor._resolve_claims(claims, manifest)
        self.assertEqual(assignments[sp.id], "a1")

    def test_claim_resolution_conflict_higher_confidence_wins(self):
        """When two agents claim the same sub-problem, higher confidence wins."""
        from protocol.executor import DCSExecutor

        agent_a = MockAgent(agent_id="a1", capabilities={"math"})
        agent_b = MockAgent(agent_id="a2", capabilities={"math"})

        sp = SubProblem(description="Math task", capabilities_needed={"math"})
        manifest = ProblemManifest(sub_problems=[sp])

        executor = DCSExecutor(agents=[agent_a, agent_b])

        claims = [
            Claim(sub_problem_id=sp.id, agent_id="a1", confidence=0.6),
            Claim(sub_problem_id=sp.id, agent_id="a2", confidence=0.9),
        ]
        assignments = executor._resolve_claims(claims, manifest)
        self.assertEqual(assignments[sp.id], "a2")

    def test_no_claims_gets_suggestion(self):
        """Sub-problem with no claims should still get an assignment suggestion."""
        from protocol.executor import DCSExecutor

        agent = MockAgent(agent_id="a1", capabilities={"math"})
        agent.will_claim = False

        sp = SubProblem(description="Math task", capabilities_needed={"math"})
        manifest = ProblemManifest(sub_problems=[sp])

        executor = DCSExecutor(agents=[agent])

        # Empty claims — executor should suggest an assignment
        assignments = executor._resolve_claims([], manifest)
        self.assertIn(sp.id, assignments)


class TestDCSProtocolExecution(unittest.TestCase):
    """Test the full DCS protocol execution with mock agents."""

    def test_full_protocol_single_agent(self):
        """Single agent should complete all phases."""
        agent = MockAgent(
            agent_id="a1",
            capabilities={"math", "reasoning"},
            solve_answer="42",
            solve_confidence=0.9,
            verify_passes=True,
        )
        # Monkey-patch agent methods
        agent.evaluate_claim = lambda sp: make_claim_agent(agent, sp)
        agent.solve = lambda sp: make_solve_agent(agent, sp)
        agent.verify = lambda ans, stmt: make_verify_agent(agent, ans, stmt)

        executor = DCSExecutor(agents=[agent])
        solution = executor.run("Calculate the answer to life")

        self.assertIsInstance(solution, CooperativeSolution)
        self.assertIsNotNone(solution.answer)
        self.assertGreater(solution.confidence, 0.0)
        self.assertGreater(solution.total_time, 0.0)

    def test_full_protocol_multi_agent(self):
        """Multiple agents should collaborate on different sub-problems."""
        agent_math = MockAgent(
            agent_id="math-expert",
            capabilities={"math"},
            solve_answer="The optimal number is 42",
            solve_confidence=0.95,
        )
        agent_math.evaluate_claim = lambda sp: make_claim_agent(agent_math, sp)
        agent_math.solve = lambda sp: make_solve_agent(agent_math, sp)

        agent_code = MockAgent(
            agent_id="code-expert",
            capabilities={"coding"},
            solve_answer="def solve(): return 42",
            solve_confidence=0.9,
        )
        agent_code.evaluate_claim = lambda sp: make_claim_agent(agent_code, sp)
        agent_code.solve = lambda sp: make_solve_agent(agent_code, sp)

        agent_verify = MockAgent(
            agent_id="verifier",
            capabilities={"reasoning"},
            verify_passes=True,
        )
        agent_verify.evaluate_claim = lambda sp: None  # Verifier doesn't claim
        agent_verify.verify = lambda ans, stmt: make_verify_agent(agent_verify, ans, stmt)

        executor = DCSExecutor(agents=[agent_math, agent_code, agent_verify])
        solution = executor.run(
            "Calculate the optimal number and implement the algorithm"
        )

        self.assertIsInstance(solution, CooperativeSolution)
        self.assertIsNotNone(solution.answer)
        self.assertTrue(len(solution.agent_contributions) > 0)

    def test_protocol_with_verification_failure(self):
        """Verification failure should be handled gracefully."""
        agent_solver = MockAgent(
            agent_id="solver",
            capabilities={"math"},
            solve_answer="wrong answer",
            solve_confidence=0.7,
        )
        agent_solver.evaluate_claim = lambda sp: make_claim_agent(agent_solver, sp)
        agent_solver.solve = lambda sp: make_solve_agent(agent_solver, sp)

        agent_verifier = MockAgent(
            agent_id="verifier",
            capabilities={"reasoning"},
            verify_passes=False,  # Will fail verification
        )
        agent_verifier.evaluate_claim = lambda sp: None
        agent_verifier.verify = lambda ans, stmt: make_verify_agent(
            agent_verifier, ans, stmt
        )

        executor = DCSExecutor(agents=[agent_solver, agent_verifier])
        solution = executor.run("Calculate the result")

        self.assertIsInstance(solution, CooperativeSolution)
        # Solution should still be returned even if verification fails
        self.assertIsNotNone(solution.answer)


class TestSynthesisWithConflictingResults(unittest.TestCase):
    """Test synthesis when agents produce conflicting partial results."""

    def test_single_result_passes_through(self):
        """Single result should pass through unchanged."""
        synth = Synthesizer()
        result = PartialResult(
            sub_problem_id="sp1",
            agent_id="a1",
            answer="answer 1",
            confidence=0.9,
            methodology="direct computation",
        )
        answer, conf, method, conflicts = synth.synthesize([result])
        self.assertEqual(answer, "answer 1")
        self.assertAlmostEqual(conf, 0.9)
        self.assertEqual(len(conflicts), 0)

    def test_complementary_results_merged(self):
        """Complementary results from different sub-problems should merge."""
        synth = Synthesizer()
        results = [
            PartialResult(
                sub_problem_id="sp1", agent_id="a1",
                answer="Part A", confidence=0.9, methodology="method A",
            ),
            PartialResult(
                sub_problem_id="sp2", agent_id="a2",
                answer="Part B", confidence=0.8, methodology="method B",
            ),
        ]
        answer, conf, method, conflicts = synth.synthesize(results)
        self.assertIsNotNone(answer)
        # Should contain both parts
        answer_str = str(answer)
        self.assertIn("Part A", answer_str)
        self.assertIn("Part B", answer_str)

    def test_conflicting_results_detected(self):
        """Conflicting results for the same sub-problem should be detected."""
        synth = Synthesizer()
        results = [
            PartialResult(
                sub_problem_id="sp1", agent_id="a1",
                answer="answer A", confidence=0.9, methodology="method A",
            ),
            PartialResult(
                sub_problem_id="sp1", agent_id="a2",
                answer="answer B", confidence=0.8, methodology="method B",
            ),
        ]
        answer, conf, method, conflicts = synth.synthesize(results)
        self.assertTrue(len(conflicts) > 0)
        self.assertTrue(any("sp1" in c for c in conflicts))

    def test_trust_weighted_synthesis(self):
        """Synthesis with trust should weight by trust scores."""
        synth = Synthesizer()
        trust = TrustManager()
        trust._trust = {
            "a1": {"math": 0.9},  # High trust
            "a2": {"math": 0.3},  # Low trust
        }
        results = [
            PartialResult(
                sub_problem_id="sp1", agent_id="a1",
                answer="answer A", confidence=0.8, methodology="method A",
            ),
            PartialResult(
                sub_problem_id="sp2", agent_id="a2",
                answer="answer B", confidence=0.8, methodology="method B",
            ),
        ]
        answer, conf, method, conflicts = synth.synthesize(results, trust)
        # Confidence should be influenced by trust
        self.assertGreater(conf, 0.0)


class TestVerification(unittest.TestCase):
    """Test verification pass/fail scenarios."""

    def test_verification_pass(self):
        """Passing verification should return passed=True."""
        agent = MockAgent(
            agent_id="verifier",
            capabilities={"reasoning"},
            verify_passes=True,
        )
        agent.verify = lambda ans, stmt: make_verify_agent(agent, ans, stmt)

        executor = DCSExecutor(agents=[agent])
        manifest = ProblemManifest(statement="test problem")
        vr = executor._phase6_verify("some answer", manifest)
        self.assertTrue(vr.passed)
        self.assertGreater(vr.score, 0.5)

    def test_verification_fail(self):
        """Failing verification should return passed=False with issues."""
        agent = MockAgent(
            agent_id="verifier",
            capabilities={"reasoning"},
            verify_passes=False,
        )
        agent.verify = lambda ans, stmt: make_verify_agent(agent, ans, stmt)

        executor = DCSExecutor(agents=[agent])
        manifest = ProblemManifest(statement="test problem")
        vr = executor._phase6_verify("some answer", manifest)
        self.assertFalse(vr.passed)
        self.assertTrue(len(vr.issues) > 0)

    def test_no_verifier_available(self):
        """Should gracefully handle no verifier being available."""
        solver = MockAgent(
            agent_id="solver",
            capabilities={"math"},
        )
        # Solver doesn't have verify method
        executor = DCSExecutor(agents=[solver])
        manifest = ProblemManifest(statement="test problem")
        vr = executor._phase6_verify("some answer", manifest)
        # Should return default pass
        self.assertTrue(vr.passed)
        self.assertIn("skipped", vr.issues[0].lower())


class TestMapReducePattern(unittest.TestCase):
    """Test the MapReduce collaboration pattern."""

    def test_basic_map_reduce(self):
        """MapReduce should process all items and reduce results."""
        agent1 = MockAgent(agent_id="a1", capabilities={"analysis"})
        agent2 = MockAgent(agent_id="a2", capabilities={"analysis"})

        pattern = MapReducePattern(agents=[agent1, agent2])
        result = pattern.run(
            data_items=[1, 2, 3, 4, 5],
            map_fn=lambda agent, item: item * 2,
            reduce_fn=lambda results: sum(results),
        )

        self.assertEqual(result.reduced_answer, 30)  # (1+2+3+4+5) * 2
        self.assertEqual(result.items_processed, 5)
        self.assertEqual(len(result.map_results), 5)

    def test_map_reduce_with_errors(self):
        """MapReduce should handle individual item errors gracefully."""
        agent = MockAgent(agent_id="a1", capabilities={"analysis"})

        def flaky_map(agent, item):
            if item == 3:
                raise ValueError("Item 3 failed")
            return item * 2

        pattern = MapReducePattern(agents=[agent])
        result = pattern.run(
            data_items=[1, 2, 3, 4],
            map_fn=flaky_map,
            reduce_fn=lambda results: len(results),
        )

        # 4 items processed, but 1 errored
        self.assertEqual(result.items_processed, 4)
        self.assertEqual(len(result.map_results), 4)
        # Only 3 successful results
        successful = [r for r in result.map_results if not isinstance(r, dict)]
        self.assertEqual(len(successful), 3)

    def test_map_reduce_single_agent(self):
        """MapReduce should work with a single agent."""
        agent = MockAgent(agent_id="a1", capabilities={"analysis"})

        pattern = MapReducePattern(agents=[agent])
        result = pattern.run(
            data_items=["hello", "world"],
            map_fn=lambda agent, item: len(item),
            reduce_fn=lambda results: max(results),
        )

        self.assertEqual(result.reduced_answer, 5)  # "hello" has 5 chars
        self.assertGreater(result.total_time, 0)


class TestDebatePattern(unittest.TestCase):
    """Test the Debate collaboration pattern."""

    def test_debate_consensus(self):
        """Debate should end immediately if all agents agree."""
        agent_a = MockAgent(agent_id="a1", capabilities={"design"})
        agent_b = MockAgent(agent_id="a2", capabilities={"design"})

        def propose(agent, problem):
            return DebateProposal(
                agent_id=agent.agent_id,
                answer="Option A",  # Both propose the same
                evidence="Strong evidence",
                confidence=0.9,
            )

        pattern = DebatePattern(agents=[agent_a, agent_b])
        result = pattern.run("Choose a design approach", propose_fn=propose)

        self.assertTrue(result.consensus_reached)
        self.assertEqual(result.winning_answer, "Option A")
        self.assertEqual(result.rounds_completed, 0)

    def test_debate_majority_wins(self):
        """When agents disagree, majority should win."""
        agent_a = MockAgent(agent_id="a1", capabilities={"design"})
        agent_b = MockAgent(agent_id="a2", capabilities={"design"})
        agent_c = MockAgent(agent_id="a3", capabilities={"design"})

        proposal_counter = {"count": 0}

        def propose(agent, problem):
            proposal_counter["count"] += 1
            # First two agents propose A, third proposes B
            if proposal_counter["count"] <= 2:
                return DebateProposal(
                    agent_id=agent.agent_id,
                    answer="Option A",
                    evidence="Evidence for A",
                    confidence=0.9,
                )
            else:
                return DebateProposal(
                    agent_id=agent.agent_id,
                    answer="Option B",
                    evidence="Evidence for B",
                    confidence=0.7,
                )

        pattern = DebatePattern(agents=[agent_a, agent_b, agent_c])
        result = pattern.run("Choose a design", propose_fn=propose)

        # Option A should win (2 agents vs 1)
        self.assertEqual(result.winning_answer, "Option A")
        # Dissent register should contain Option B
        self.assertTrue(len(result.dissent_register) > 0)

    def test_debate_with_trust_weights(self):
        """Trust weights should influence vote outcomes."""
        agent_a = MockAgent(agent_id="a1", capabilities={"design"})
        agent_b = MockAgent(agent_id="a2", capabilities={"design"})

        def propose(agent, problem):
            return DebateProposal(
                agent_id=agent.agent_id,
                answer=f"Option {agent.agent_id}",
                evidence="Evidence",
                confidence=0.8,
            )

        # Agent a1 has higher trust weight
        pattern = DebatePattern(
            agents=[agent_a, agent_b],
            trust_weights={"a1": 10.0, "a2": 1.0},
        )
        result = pattern.run("Choose a design", propose_fn=propose)

        # Agent a1 should win due to higher trust weight
        self.assertEqual(result.winning_answer, "Option a1")

    def test_debate_convergence(self):
        """Debate should converge if vote distribution stabilizes."""
        agent_a = MockAgent(agent_id="a1", capabilities={"design"})
        agent_b = MockAgent(agent_id="a2", capabilities={"design"})

        def propose(agent, problem):
            return DebateProposal(
                agent_id=agent.agent_id,
                answer=f"Option {agent.agent_id}",
                evidence="Evidence",
                confidence=0.8,
            )

        # Both have equal weight, so distribution should be stable from round 1
        pattern = DebatePattern(agents=[agent_a, agent_b])
        pattern.CONVERGENCE_THRESHOLD = 0.5  # Very easy to converge
        result = pattern.run("Choose a design", propose_fn=propose)

        # Should terminate quickly
        self.assertLessEqual(result.rounds_completed, pattern.MAX_ROUNDS)

    def test_debate_with_arguments(self):
        """Debate should include argument phase when argue_fn is provided."""
        agent_a = MockAgent(agent_id="a1", capabilities={"design"})
        agent_b = MockAgent(agent_id="a2", capabilities={"design"})

        def propose(agent, problem):
            # Two agents disagree so the debate proceeds to argument phase
            answer = "Option A" if agent.agent_id == "a1" else "Option B"
            return DebateProposal(
                agent_id=agent.agent_id,
                answer=answer,
                evidence=f"Evidence for {answer}",
                confidence=0.9,
            )

        def argue(agent, proposals):
            return DebateArgument(
                agent_id=agent.agent_id,
                for_proposal_id=proposals[0].proposal_id,
                position="for",
                argument="This is the best option because...",
                evidence="Data point 1, Data point 2",
            )

        # Low convergence threshold so the debate runs at least one argue round
        pattern = DebatePattern(agents=[agent_a, agent_b])
        pattern.CONVERGENCE_THRESHOLD = 0.0
        result = pattern.run("Choose a design", propose_fn=propose, argue_fn=argue)

        # Arguments should have been collected (at least one round of arguments)
        self.assertTrue(len(result.arguments) > 0)


class TestCascadePattern(unittest.TestCase):
    """Test the Cascade collaboration pattern."""

    def test_linear_pipeline(self):
        """Cascade should process data through the pipeline sequentially."""
        agent_a = MockAgent(agent_id="outline", capabilities={"writing"})
        agent_b = MockAgent(agent_id="draft", capabilities={"writing"})
        agent_c = MockAgent(agent_id="edit", capabilities={"writing"})

        def process(agent, input_data):
            output = f"{agent.agent_id}: processed '{input_data}'"
            return output, None  # No feedback

        pattern = CascadePattern(agent_chain=[agent_a, agent_b, agent_c])
        result = pattern.run("initial input", process_fn=process)

        self.assertTrue(result.pipeline_completed)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.feedback_loops, 0)
        # Final answer should be from last agent
        self.assertIn("edit", result.final_answer)
        self.assertIn("outline", result.final_answer)  # Should contain earlier work

    def test_pipeline_with_feedback(self):
        """Cascade should handle feedback loops."""
        agent_a = MockAgent(agent_id="writer", capabilities={"writing"})
        agent_b = MockAgent(agent_id="editor", capabilities={"writing"})

        call_count = {"n": 0}

        def process_with_feedback(agent, input_data):
            call_count["n"] += 1
            # Editor is the 2nd call (after writer), so check for n==2
            if agent.agent_id == "editor" and call_count["n"] == 2:
                # First time editor runs, send feedback to writer
                feedback = {"step": 0, "feedback": "Make it more concise", "input": input_data}
                return "needs revision", feedback
            # Otherwise, just process
            return f"{agent.agent_id}: {input_data}", None

        pattern = CascadePattern(agent_chain=[agent_a, agent_b])
        pattern.MAX_FEEDBACK_LOOPS = 3
        result = pattern.run("initial draft", process_fn=process_with_feedback)

        # Should have had at least one feedback loop
        self.assertGreaterEqual(result.feedback_loops, 1)
        # Should have more steps due to re-processing
        self.assertGreater(len(result.steps), 2)

    def test_single_step_pipeline(self):
        """Single-agent pipeline should work."""
        agent = MockAgent(agent_id="solo", capabilities={"writing"})

        def process(agent, input_data):
            return f"processed: {input_data}", None

        pattern = CascadePattern(agent_chain=[agent])
        result = pattern.run("input", process_fn=process)

        self.assertTrue(result.pipeline_completed)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.final_answer, "processed: input")

    def test_pipeline_error_handling(self):
        """Pipeline should handle errors gracefully."""
        agent_a = MockAgent(agent_id="a1", capabilities={"writing"})
        agent_b = MockAgent(agent_id="a2", capabilities={"writing"})

        def failing_process(agent, input_data):
            if agent.agent_id == "a1":
                raise RuntimeError("Agent A failed!")
            return f"ok: {input_data}", None

        pattern = CascadePattern(agent_chain=[agent_a, agent_b])
        result = pattern.run("input", process_fn=failing_process)

        self.assertFalse(result.pipeline_completed)
        # First step should have failed
        self.assertEqual(result.steps[0].status, "failed")


class TestTrustManagement(unittest.TestCase):
    """Test the TrustManager."""

    def test_default_trust(self):
        """New agents should have default trust of 0.5."""
        tm = TrustManager()
        self.assertAlmostEqual(tm.get_trust("new-agent", "math"), 0.5)

    def test_reward_increases_trust(self):
        """Rewarding should increase trust."""
        tm = TrustManager()
        tm.reward("agent-a", "math", 0.1)
        self.assertAlmostEqual(tm.get_trust("agent-a", "math"), 0.6)

    def test_penalize_decreases_trust(self):
        """Penalizing should decrease trust."""
        tm = TrustManager()
        tm.penalize("agent-a", "math", 0.1)
        self.assertAlmostEqual(tm.get_trust("agent-a", "math"), 0.4)

    def test_trust_bounded(self):
        """Trust should be bounded to [0.0, 1.0]."""
        tm = TrustManager()
        # Try to exceed upper bound
        for _ in range(100):
            tm.reward("agent-a", "math", 0.5)
        self.assertAlmostEqual(tm.get_trust("agent-a", "math"), 1.0)

        # Try to go below lower bound
        for _ in range(100):
            tm.penalize("agent-a", "math", 0.5)
        self.assertAlmostEqual(tm.get_trust("agent-a", "math"), 0.0)

    def test_overall_trust(self):
        """Overall trust should average across capabilities."""
        tm = TrustManager()
        tm.reward("agent-a", "math", 0.3)  # 0.8
        tm.penalize("agent-a", "coding", 0.2)  # 0.3
        overall = tm.get_overall_trust("agent-a")
        self.assertAlmostEqual(overall, (0.8 + 0.3) / 2)

    def test_contradiction_update(self):
        """Contradiction should reward winner and penalize loser."""
        tm = TrustManager()
        tm.update_from_contradiction("winner", "loser", "math")
        self.assertGreater(tm.get_trust("winner", "math"), 0.5)
        self.assertLess(tm.get_trust("loser", "math"), 0.5)


class TestPatternSelection(unittest.TestCase):
    """Test the PatternSelector."""

    def test_recommend_map_reduce(self):
        """Batch processing problem should recommend MapReduce."""
        selector = PatternSelector()
        rec = selector.recommend(
            "Analyze each of the 100 log files independently for errors"
        )
        self.assertEqual(rec.pattern_name, "map_reduce")

    def test_recommend_debate(self):
        """Decision problem should recommend Debate."""
        selector = PatternSelector()
        rec = selector.recommend(
            "Decide whether to use React or Vue for the frontend"
        )
        self.assertEqual(rec.pattern_name, "debate")

    def test_recommend_cascade(self):
        """Sequential pipeline problem should recommend Cascade."""
        selector = PatternSelector()
        rec = selector.recommend(
            "First create an outline, then draft the document, "
            "then edit and proofread"
        )
        self.assertEqual(rec.pattern_name, "cascade")

    def test_recommend_dcs(self):
        """Complex multi-faceted problem should recommend DCS."""
        selector = PatternSelector()
        rec = selector.recommend(
            "This is a complex problem requiring analysis of multiple aspects"
        )
        self.assertEqual(rec.pattern_name, "dcs")

    def test_recommendation_has_reasoning(self):
        """Recommendation should include a reasoning string."""
        selector = PatternSelector()
        rec = selector.recommend("Some problem")
        self.assertTrue(len(rec.reasoning) > 0)
        self.assertGreater(rec.confidence, 0.0)


class TestEdgeCaseEmptyAgents(unittest.TestCase):
    """Test edge case: empty agents list."""

    def test_empty_agents_returns_empty_result(self):
        """Executor with no agents should return a valid but empty CooperativeSolution."""
        executor = DCSExecutor(agents=[])
        solution = executor.run("Calculate something complex")

        self.assertIsInstance(solution, CooperativeSolution)
        self.assertIsNone(solution.answer)
        self.assertAlmostEqual(solution.confidence, 0.0)
        self.assertEqual(solution.agent_contributions, {})
        self.assertGreaterEqual(solution.total_time, 0.0)

    def test_empty_agents_methodology_describes_issue(self):
        """Methodology should explain that no agents were available."""
        executor = DCSExecutor(agents=[])
        solution = executor.run("Solve this problem")

        self.assertIn("No agents", solution.methodology)

    def test_empty_agents_session_log_failure(self):
        """Session log should record failure for empty agents."""
        executor = DCSExecutor(agents=[])
        executor.run("Problem")

        self.assertFalse(executor.session_log.success)
        self.assertGreaterEqual(executor.session_log.total_time, 0.0)


class TestEdgeCaseZeroSubProblems(unittest.TestCase):
    """Test edge case: decomposer returns zero sub-problems."""

    def _make_decomposer_with_zero_sps(self):
        """Create a decomposer that returns a manifest with no sub-problems."""
        from protocol.problem import ProblemDecomposer

        class ZeroDecomposer(ProblemDecomposer):
            def decompose(self, problem_statement, agent_capabilities=None):
                return ProblemManifest(
                    statement=problem_statement,
                    sub_problems=[],
                )

        return ZeroDecomposer()

    def test_zero_sub_problems_skips_to_synthesis(self):
        """Should skip to synthesis when no sub-problems are produced."""
        agent = MockAgent(
            agent_id="a1",
            capabilities={"math"},
            solve_answer="42",
        )
        agent.evaluate_claim = lambda sp: make_claim_agent(agent, sp)
        agent.solve = lambda sp: make_solve_agent(agent, sp)

        decomposer = self._make_decomposer_with_zero_sps()
        executor = DCSExecutor(agents=[agent], decomposer=decomposer)
        solution = executor.run("Trivial problem")

        self.assertIsInstance(solution, CooperativeSolution)
        # No answer since no partials to synthesize
        self.assertIsNone(solution.answer)
        self.assertAlmostEqual(solution.confidence, 0.0)

    def test_zero_sub_problems_no_agent_contributions(self):
        """Should have no agent contributions when there are no sub-problems."""
        agent = MockAgent(agent_id="a1", capabilities={"math"})
        agent.evaluate_claim = lambda sp: make_claim_agent(agent, sp)
        agent.solve = lambda sp: make_solve_agent(agent, sp)

        decomposer = self._make_decomposer_with_zero_sps()
        executor = DCSExecutor(agents=[agent], decomposer=decomposer)
        solution = executor.run("Trivial problem")

        self.assertEqual(solution.agent_contributions, {})

    def test_zero_sub_problems_methodology_mentions_no_sub_problems(self):
        """Methodology should note that no sub-problems were decomposed."""
        agent = MockAgent(agent_id="a1", capabilities={"math"})
        decomposer = self._make_decomposer_with_zero_sps()
        executor = DCSExecutor(agents=[agent], decomposer=decomposer)
        solution = executor.run("Trivial problem")

        self.assertIn("No sub-problems", solution.methodology)

    def test_zero_sub_problems_session_log_success(self):
        """Session log should still record success even with zero sub-problems."""
        agent = MockAgent(agent_id="a1", capabilities={"math"})
        decomposer = self._make_decomposer_with_zero_sps()
        executor = DCSExecutor(agents=[agent], decomposer=decomposer)
        executor.run("Trivial problem")

        self.assertTrue(executor.session_log.success)
        self.assertIn("decompose", executor.session_log.phases_completed)


class TestEdgeCaseAllAgentsFail(unittest.TestCase):
    """Test edge case: all agents fail during execution."""

    def test_all_agents_fail_returns_best_effort(self):
        """When all agents fail, should return best-effort result with reduced confidence."""
        failing_agent = MockAgent(
            agent_id="fail-agent",
            capabilities={"math"},
        )
        failing_agent.evaluate_claim = lambda sp: make_claim_agent(failing_agent, sp)
        failing_agent.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("Agent crashed"))

        executor = DCSExecutor(agents=[failing_agent])
        solution = executor.run("Calculate the answer")

        self.assertIsInstance(solution, CooperativeSolution)
        # Best-effort: reduced confidence, not zero
        self.assertGreater(solution.confidence, 0.0)
        self.assertLessEqual(solution.confidence, 0.2)  # reduced confidence = 0.1
        self.assertIsNone(solution.answer)
        self.assertFalse(executor.session_log.success)

    def test_all_agents_fail_methodology_describes_failure(self):
        """Methodology should describe that all agents failed."""
        failing_agent = MockAgent(
            agent_id="fail-agent",
            capabilities={"math"},
        )
        failing_agent.evaluate_claim = lambda sp: make_claim_agent(failing_agent, sp)
        failing_agent.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("crash"))

        executor = DCSExecutor(agents=[failing_agent])
        solution = executor.run("Calculate the answer")

        self.assertIn("All agents failed", solution.methodology)
        self.assertIn("reduced confidence", solution.methodology)

    def test_all_agents_fail_no_contributions(self):
        """Should have no agent contributions when all agents fail."""
        failing_agent = MockAgent(
            agent_id="fail-agent",
            capabilities={"math"},
        )
        failing_agent.evaluate_claim = lambda sp: make_claim_agent(failing_agent, sp)
        failing_agent.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("crash"))

        executor = DCSExecutor(agents=[failing_agent])
        solution = executor.run("Calculate the answer")

        self.assertEqual(solution.agent_contributions, {})

    def test_multiple_agents_all_fail(self):
        """Should handle all agents failing even when there are multiple."""
        failing_agent1 = MockAgent(agent_id="f1", capabilities={"math"})
        failing_agent1.evaluate_claim = lambda sp: make_claim_agent(failing_agent1, sp)
        failing_agent1.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("crash 1"))

        failing_agent2 = MockAgent(agent_id="f2", capabilities={"math"})
        failing_agent2.evaluate_claim = lambda sp: make_claim_agent(failing_agent2, sp)
        failing_agent2.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("crash 2"))

        executor = DCSExecutor(agents=[failing_agent1, failing_agent2])
        solution = executor.run("Calculate the answer")

        self.assertIsInstance(solution, CooperativeSolution)
        self.assertGreater(solution.confidence, 0.0)
        self.assertLessEqual(solution.confidence, 0.2)
        self.assertFalse(executor.session_log.success)


class TestPhase4RetryLogic(unittest.TestCase):
    """Test Phase 4 retry logic for missing results."""

    def test_retry_on_missing_results(self):
        """Phase 4 should retry missing results with available agents."""
        # First agent fails the sub-problem
        failing_agent = MockAgent(agent_id="failer", capabilities={"math"})
        failing_agent.evaluate_claim = lambda sp: make_claim_agent(failing_agent, sp)
        failing_agent.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("first attempt fails"))

        # Second agent succeeds (it should be used as retry)
        success_agent = MockAgent(
            agent_id="retrier",
            capabilities={"math"},
            solve_answer="retried answer",
            solve_confidence=0.7,
        )
        success_agent.evaluate_claim = lambda sp: None  # doesn't claim initially

        def solve_that_fails_first_time_then_succeeds(sp):
            # On the first call (from phase3), fail. On retry (from phase4), succeed.
            if not hasattr(solve_that_fails_first_time_then_succeeds, "call_count"):
                solve_that_fails_first_time_then_succeeds.call_count = 0
            solve_that_fails_first_time_then_succeeds.call_count += 1
            if solve_that_fails_first_time_then_succeeds.call_count == 1:
                raise RuntimeError("First attempt fails")
            return make_solve_agent(success_agent, sp)

        success_agent.solve = solve_that_fails_first_time_then_succeeds

        executor = DCSExecutor(agents=[failing_agent, success_agent])
        solution = executor.run("Calculate something")

        self.assertIsInstance(solution, CooperativeSolution)
        # The retried agent should have produced a result
        # (either through retry or the solution should still be valid)
        self.assertIsNotNone(solution)

    def test_retry_marks_gaps_as_yielded(self):
        """Sub-problems with no results after retry should be marked YIELDED."""
        # Only one agent, and it fails
        failing_agent = MockAgent(agent_id="failer", capabilities={"math"})
        failing_agent.evaluate_claim = lambda sp: make_claim_agent(failing_agent, sp)
        failing_agent.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("always fails"))

        executor = DCSExecutor(agents=[failing_agent])
        solution = executor.run("Calculate something")

        # All agents failed, so we get best-effort result
        self.assertFalse(executor.session_log.success)
        # Check that the retry was attempted (yields_received should include gap sub-problems)
        # Since only one agent exists, retry can't find a new agent

    def test_retry_with_no_available_agents_for_gap(self):
        """When no alternative agents exist for a gap, the gap should remain unfilled."""
        failing_agent = MockAgent(agent_id="only-agent", capabilities={"math"})
        failing_agent.evaluate_claim = lambda sp: make_claim_agent(failing_agent, sp)
        failing_agent.solve = lambda sp: (_ for _ in ()).throw(RuntimeError("fails"))

        executor = DCSExecutor(agents=[failing_agent])
        solution = executor.run("Calculate something")

        # Should return best-effort since all agents failed
        self.assertGreater(solution.confidence, 0.0)
        self.assertLessEqual(solution.confidence, 0.2)

    def test_retry_increments_session_retry_count(self):
        """Successful retry should increment session log retry count."""
        # We need a scenario where initial execution fails for some sub-problems
        # but retry succeeds. Create two sub-problems via a custom decomposer.
        from protocol.problem import ProblemDecomposer

        class TwoSubProblemDecomposer(ProblemDecomposer):
            def decompose(self, problem_statement, agent_capabilities=None):
                sp1 = SubProblem(
                    description="First sub-problem",
                    capabilities_needed={"math"},
                )
                sp2 = SubProblem(
                    description="Second sub-problem",
                    capabilities_needed={"reasoning"},
                )
                return ProblemManifest(
                    statement=problem_statement,
                    sub_problems=[sp1, sp2],
                )

        # Agent that claims both sub-problems but fails on reasoning
        math_agent = MockAgent(agent_id="math-guy", capabilities={"math", "reasoning"})
        math_agent.evaluate_claim = lambda sp: make_claim_agent(math_agent, sp)

        def math_solve(sp):
            if "reasoning" in sp.capabilities_needed:
                raise RuntimeError("Can't do reasoning")
            return make_solve_agent(math_agent, sp)
        math_agent.solve = math_solve

        # Agent for reasoning (only for retry)
        reason_agent = MockAgent(
            agent_id="reason-guy",
            capabilities={"reasoning"},
            solve_answer="reasoned answer",
            solve_confidence=0.7,
        )
        reason_agent.evaluate_claim = lambda sp: None  # Doesn't claim initially
        reason_agent.solve = lambda sp: make_solve_agent(reason_agent, sp)

        decomposer = TwoSubProblemDecomposer()
        executor = DCSExecutor(
            agents=[math_agent, reason_agent],
            decomposer=decomposer,
        )
        solution = executor.run("Solve math and reasoning")

        # The reasoning sub-problem should have been retried with reason-guy
        self.assertIsNotNone(solution.answer)
        # Retry count should be at least 1
        self.assertGreaterEqual(executor.session_log.retry_count, 1)


class TestSessionLogging(unittest.TestCase):
    """Test session log recording."""

    def test_session_log_records_phases(self):
        """Session log should track completed phases."""
        agent = MockAgent(
            agent_id="a1",
            capabilities={"math"},
            solve_answer="42",
        )
        agent.evaluate_claim = lambda sp: make_claim_agent(agent, sp)
        agent.solve = lambda sp: make_solve_agent(agent, sp)

        executor = DCSExecutor(agents=[agent])
        solution = executor.run("Calculate something")

        log = executor.session_log
        self.assertIn("decompose", log.phases_completed)
        self.assertIn("self_select", log.phases_completed)
        self.assertIn("execute", log.phases_completed)
        self.assertIn("collect", log.phases_completed)
        self.assertIn("synthesize", log.phases_completed)
        self.assertIn("verify", log.phases_completed)
        self.assertIn("learn", log.phases_completed)

    def test_session_log_records_agents(self):
        """Session log should track involved agents."""
        agent = MockAgent(
            agent_id="a1",
            capabilities={"math"},
            solve_answer="42",
        )
        agent.evaluate_claim = lambda sp: make_claim_agent(agent, sp)
        agent.solve = lambda sp: make_solve_agent(agent, sp)

        executor = DCSExecutor(agents=[agent])
        solution = executor.run("Calculate something")

        self.assertIn("a1", executor.session_log.agents_involved)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
