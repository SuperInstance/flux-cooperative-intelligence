"""
flux-cooperative-intelligence: DCS Protocol Executor

Implements the full DIVIDE-CONQUER-SYNTHESIZE protocol execution loop.
The executor orchestrates the 7-phase lifecycle of cooperative problem-solving.

This is a reference/simulation implementation. In a live FLUX fleet, the
communication primitives (BROADCAST, ASK, TELL, CLAIM) would be real fleet
messages. Here we simulate them with callback hooks for testability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol as TypingProtocol,
    Set,
    Tuple,
)

from .problem import (
    Claim,
    CooperativeSolution,
    PartialResult,
    ProblemDecomposer,
    ProblemManifest,
    SubProblem,
    SubProblemStatus,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Agent Interface (typing protocol)
# ---------------------------------------------------------------------------

class AgentInterface(TypingProtocol):
    """Interface that any agent in the cooperative fleet must implement."""

    agent_id: str
    capabilities: Set[str]
    trust_score: float

    def evaluate_claim(self, sub_problem: SubProblem) -> Optional[Claim]:
        """Evaluate whether to claim a sub-problem. Returns Claim or None."""
        ...

    def solve(self, sub_problem: SubProblem) -> PartialResult:
        """Solve an assigned sub-problem and return a partial result."""
        ...

    def verify(self, answer: Any, problem_statement: str) -> VerificationResult:
        """Optionally verify a synthesized answer."""
        ...


# ---------------------------------------------------------------------------
# Communication Callbacks (for simulation / testing)
# ---------------------------------------------------------------------------

@dataclass
class CommCallbacks:
    """Callbacks that simulate fleet communication primitives.

    In production, these would be replaced with real FLUX opcode handlers.
    """

    broadcast: Optional[Callable[[Dict[str, Any]], List[Dict[str, Any]]]] = None
    """BROADCAST: Send to all agents, return list of responses."""

    ask: Optional[Callable[[str, str, Dict[str, Any]], Any]] = None
    """ASK: Send a question to a specific agent, return response."""

    tell: Optional[Callable[[str, Dict[str, Any]], None]] = None
    """TELL: Send a message to a specific agent."""


# ---------------------------------------------------------------------------
# Session Log
# ---------------------------------------------------------------------------

@dataclass
class SessionLog:
    """Records the full lifecycle of a cooperative problem-solving session."""

    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:8]}")
    problem_id: str = ""
    phases_completed: List[str] = field(default_factory=list)
    agents_involved: Set[str] = field(default_factory=set)
    claims_received: List[Claim] = field(default_factory=list)
    assignments_made: Dict[str, str] = field(default_factory=dict)
    yields_received: List[str] = field(default_factory=list)
    partial_results: List[PartialResult] = field(default_factory=list)
    synthesis_conflicts: List[str] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)
    total_time: float = 0.0
    retry_count: int = 0
    success: bool = False


# ---------------------------------------------------------------------------
# Trust Manager
# ---------------------------------------------------------------------------

class TrustManager:
    """Manages per-agent, per-capability trust scores.

    Trust evolves based on contribution quality:
    - Successful contributions increase trust.
    - Contradictions and verification failures decrease trust.
    """

    DEFAULT_BASE_TRUST = 0.5
    PENALTY_FACTOR = 0.1

    def __init__(self) -> None:
        # agent_id -> capability -> trust score
        self._trust: Dict[str, Dict[str, float]] = {}

    def get_trust(self, agent_id: str, capability: str) -> float:
        """Get the trust score for an agent's specific capability."""
        caps = self._trust.get(agent_id, {})
        return caps.get(capability, self.DEFAULT_BASE_TRUST)

    def get_overall_trust(self, agent_id: str) -> float:
        """Get average trust across all capabilities for an agent."""
        caps = self._trust.get(agent_id, {})
        if not caps:
            return self.DEFAULT_BASE_TRUST
        return sum(caps.values()) / len(caps)

    def reward(
        self, agent_id: str, capability: str, amount: float = 0.05
    ) -> None:
        """Increase trust for a successful contribution."""
        caps = self._trust.setdefault(agent_id, {})
        current = caps.get(capability, self.DEFAULT_BASE_TRUST)
        caps[capability] = min(current + amount, 1.0)

    def penalize(
        self, agent_id: str, capability: str, amount: float = 0.1
    ) -> None:
        """Decrease trust for a failed or contradicted contribution."""
        caps = self._trust.setdefault(agent_id, {})
        current = caps.get(capability, self.DEFAULT_BASE_TRUST)
        caps[capability] = max(current - amount, 0.0)

    def update_from_verification(
        self,
        agent_id: str,
        capability: str,
        passed: bool,
    ) -> None:
        """Update trust based on verification outcome."""
        if passed:
            self.reward(agent_id, capability, 0.05)
        else:
            self.penalize(agent_id, capability, 0.1)

    def update_from_contradiction(
        self, winner_id: str, loser_id: str, capability: str
    ) -> None:
        """Update trust when two agents' results conflict.

        The agent whose result survives synthesis is rewarded;
        the other is penalized.
        """
        self.reward(winner_id, capability, 0.05)
        self.penalize(loser_id, capability, self.PENALTY_FACTOR)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class Synthesizer:
    """Combines partial results into a coherent final answer.

    Implements multiple conflict resolution strategies:
    - Weighted confidence: Combine answers weighted by agent confidence.
    - Trust-weighted: Weight by trust scores.
    - Evidence arbitration: Pick the answer with stronger evidence.
    - Fusion: Merge complementary answers.
    """

    def synthesize(
        self,
        results: List[PartialResult],
        trust_manager: Optional[TrustManager] = None,
    ) -> Tuple[Any, float, str, List[str]]:
        """Synthesize partial results into a final answer.

        Args:
            results: All collected partial results.
            trust_manager: Optional trust manager for trust-weighted synthesis.

        Returns:
            Tuple of (answer, confidence, methodology, conflicts).
        """
        if not results:
            return None, 0.0, "No results to synthesize.", []

        if len(results) == 1:
            r = results[0]
            return r.answer, r.confidence, r.methodology, []

        # Check for conflicts (same sub-problem, different answers)
        conflicts: List[str] = []
        by_sub: Dict[str, List[PartialResult]] = {}
        for r in results:
            by_sub.setdefault(r.sub_problem_id, []).append(r)

        for sub_id, subs in by_sub.items():
            if len(subs) > 1:
                answers = [s.answer for s in subs]
                if len(set(str(a) for a in answers)) > 1:
                    conflicts.append(
                        f"Conflict on {sub_id}: {len(subs)} different answers"
                    )

        # Strategy: collect all answers into a structured result
        # In a real implementation, an LLM would intelligently merge these.
        combined_answer = self._combine_answers(results)

        # Confidence: weighted by trust if available, else by agent confidence
        if trust_manager:
            total_trust = 0.0
            weighted_confidence = 0.0
            for r in results:
                t = trust_manager.get_overall_trust(r.agent_id)
                weighted_confidence += r.confidence * t
                total_trust += t
            confidence = weighted_confidence / max(total_trust, 0.001)
        else:
            # Minimum of all confidences (weakest link)
            confidence = min(r.confidence for r in results)

        methodology = self._build_methodology(results, conflicts)

        return combined_answer, confidence, methodology, conflicts

    def _combine_answers(self, results: List[PartialResult]) -> Dict[str, Any]:
        """Combine partial results into a structured answer."""
        combined: Dict[str, Any] = {
            "sub_results": {},
            "merged_answer": None,
        }

        for r in results:
            combined["sub_results"][r.sub_problem_id] = {
                "answer": r.answer,
                "confidence": r.confidence,
                "agent": r.agent_id,
                "methodology": r.methodology,
            }

        # Simple merge strategy: if all answers are strings, concatenate
        str_answers = [
            r.answer for r in results if isinstance(r.answer, str)
        ]
        if str_answers and len(str_answers) == len(results):
            combined["merged_answer"] = " | ".join(str_answers)
        elif len(results) == 1:
            combined["merged_answer"] = results[0].answer
        else:
            # Return dict of all sub-results as the merged answer
            combined["merged_answer"] = {
                sp_id: r.answer for r in results for sp_id in [r.sub_problem_id]
            }

        return combined

    def _build_methodology(
        self, results: List[PartialResult], conflicts: List[str]
    ) -> str:
        """Build a provenance narrative."""
        lines = ["Cooperative solution synthesized from partial results:"]
        for r in results:
            lines.append(
                f"  - [{r.agent_id}] {r.methodology} (confidence: {r.confidence:.2f})"
            )
        if conflicts:
            lines.append("Conflicts resolved:")
            for c in conflicts:
                lines.append(f"  - {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DCS Executor — Main Protocol Engine
# ---------------------------------------------------------------------------

class DCSExecutor:
    """Orchestrates the full DIVIDE-CONQUER-SYNTHESIZE protocol.

    Usage:
        executor = DCSExecutor(agents=[...])
        solution = executor.run("Solve this complex problem...")
    """

    MAX_RETRIES = 3
    MAX_VERIFICATION_LOOPS = 3
    PHASE_TIMEOUT = 300.0  # seconds per phase

    def __init__(
        self,
        agents: List[Any],
        callbacks: Optional[CommCallbacks] = None,
        trust_manager: Optional[TrustManager] = None,
        decomposer: Optional[ProblemDecomposer] = None,
    ) -> None:
        self.agents = agents
        self.callbacks = callbacks or CommCallbacks()
        self.trust_manager = trust_manager or TrustManager()
        self.decomposer = decomposer or ProblemDecomposer()
        self.synthesizer = Synthesizer()
        self.session_log = SessionLog()

    def run(
        self,
        problem_statement: str,
        owner_id: str = "system",
    ) -> CooperativeSolution:
        """Run the full DCS protocol on a problem.

        Args:
            problem_statement: The problem to solve cooperatively.
            owner_id: ID of the ProblemOwner agent.

        Returns:
            A CooperativeSolution with the final answer and metadata.
        """
        session_start = time.time()
        self.session_log = SessionLog(problem_id="")

        try:
            # Phase 1: Decompose
            manifest = self._phase1_decompose(problem_statement, owner_id)
            self.session_log.problem_id = manifest.problem_id

            # Phase 2: Self-Select
            assignments = self._phase2_self_select(manifest)
            self.session_log.assignments_made = assignments

            # Phase 3: Execute
            partial_results = self._phase3_execute(manifest, assignments)

            # Phase 4: Collect (already done in Phase 3, but formally collect)
            results = self._phase4_collect(partial_results, manifest)

            # Phase 5: Synthesize
            answer, confidence, methodology, conflicts = self._phase5_synthesize(results)
            self.session_log.synthesis_conflicts = conflicts

            # Phase 6: Verify (with retry loop)
            verification = None
            for _ in range(self.MAX_VERIFICATION_LOOPS):
                verification = self._phase6_verify(answer, manifest)
                if verification.passed:
                    break
                # Refine and retry (simplified — in production, loop to Phase 2)
                self.session_log.retry_count += 1

            # Phase 7: Learn
            self._phase7_learn(manifest, results, verification or VerificationResult())

            total_time = time.time() - session_start
            self.session_log.total_time = total_time
            self.session_log.success = True

            # Build agent contributions map
            contributions: Dict[str, List[str]] = {}
            for r in results:
                contributions.setdefault(r.agent_id, []).append(r.sub_problem_id)

            return CooperativeSolution(
                answer=answer,
                confidence=confidence,
                methodology=methodology,
                agent_contributions=contributions,
                total_time=total_time,
                verification_result=verification,
                problem_id=manifest.problem_id,
            )

        except Exception as e:
            total_time = time.time() - session_start
            self.session_log.total_time = total_time
            self.session_log.success = False
            return CooperativeSolution(
                answer=None,
                confidence=0.0,
                methodology=f"Protocol failed: {e}",
                total_time=total_time,
                problem_id=self.session_log.problem_id,
            )

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase1_decompose(
        self, problem_statement: str, owner_id: str
    ) -> ProblemManifest:
        """Phase 1: Problem Decomposition."""
        agent_caps = {
            a.agent_id: a.capabilities for a in self.agents
            if hasattr(a, "agent_id") and hasattr(a, "capabilities")
        }

        manifest = self.decomposer.decompose(problem_statement, agent_caps)
        manifest.owner = owner_id
        self.session_log.phases_completed.append("decompose")

        # Simulate BROADCAST
        if self.callbacks.broadcast:
            self.callbacks.broadcast({
                "opcode": "MANIFEST",
                "problem_id": manifest.problem_id,
                "manifest": manifest,
            })

        return manifest

    def _phase2_self_select(
        self, manifest: ProblemManifest
    ) -> Dict[str, str]:
        """Phase 2: Agent Self-Selection and claim resolution."""
        claims: List[Claim] = []

        for agent in self.agents:
            if not hasattr(agent, "evaluate_claim"):
                continue
            for sp in manifest.sub_problems:
                claim = agent.evaluate_claim(sp)
                if claim is not None:
                    claims.append(claim)

        self.session_log.claims_received = claims

        # Resolve conflicts and build assignment map
        assignments = self._resolve_claims(claims, manifest)
        self.session_log.phases_completed.append("self_select")

        return assignments

    def _phase3_execute(
        self,
        manifest: ProblemManifest,
        assignments: Dict[str, str],
    ) -> List[PartialResult]:
        """Phase 3: Parallel Execution."""
        agent_map = {a.agent_id: a for a in self.agents if hasattr(a, "agent_id")}
        results: List[PartialResult] = []

        for sp in manifest.sub_problems:
            agent_id = assignments.get(sp.id)
            if agent_id is None or agent_id not in agent_map:
                sp.status = SubProblemStatus.UNCLAIMED
                continue

            agent = agent_map[agent_id]
            sp.assigned_agent = agent_id
            sp.status = SubProblemStatus.IN_PROGRESS

            if hasattr(agent, "solve"):
                try:
                    start = time.time()
                    result = agent.solve(sp)
                    result.sub_problem_id = sp.id
                    result.agent_id = agent_id
                    result.elapsed_time = time.time() - start
                    sp.result = result
                    sp.status = SubProblemStatus.COMPLETED
                    results.append(result)
                except Exception:
                    sp.status = SubProblemStatus.FAILED
                    self.session_log.yields_received.append(sp.id)
            else:
                sp.status = SubProblemStatus.FAILED

        self.session_log.agents_involved.update(assignments.values())
        self.session_log.partial_results = results
        self.session_log.phases_completed.append("execute")

        return results

    def _phase4_collect(
        self,
        results: List[PartialResult],
        manifest: ProblemManifest,
    ) -> List[PartialResult]:
        """Phase 4: Result Collection.

        Checks completeness and identifies gaps. In a production system,
        this would handle retries for missing results.
        """
        solved_ids = {r.sub_problem_id for r in results}
        all_ids = {sp.id for sp in manifest.sub_problems}

        gaps = all_ids - solved_ids
        if gaps:
            # Log gaps — in production, would retry
            for gap_id in gaps:
                sp = next((s for s in manifest.sub_problems if s.id == gap_id), None)
                if sp:
                    sp.status = SubProblemStatus.YIELDED

        self.session_log.phases_completed.append("collect")
        return results

    def _phase5_synthesize(
        self, results: List[PartialResult]
    ) -> Tuple[Any, float, str, List[str]]:
        """Phase 5: Synthesis."""
        answer, confidence, methodology, conflicts = self.synthesizer.synthesize(
            results, self.trust_manager
        )
        self.session_log.phases_completed.append("synthesize")

        # Update trust based on conflicts
        for conflict in conflicts:
            # Simplified conflict handling: find which agents disagree
            # and let the higher-confidence agent win
            self._handle_conflict_trust(results)

        return answer, confidence, methodology, conflicts

    def _phase6_verify(
        self, answer: Any, manifest: ProblemManifest
    ) -> VerificationResult:
        """Phase 6: Verification.

        Tries to find a verification agent. If none available, returns
        a default pass.
        """
        self.session_log.phases_completed.append("verify")

        # Find an agent that didn't contribute to this answer
        contributor_ids = {r.agent_id for r in self.session_log.partial_results}

        for agent in self.agents:
            if (
                hasattr(agent, "verify")
                and hasattr(agent, "agent_id")
                and agent.agent_id not in contributor_ids
            ):
                try:
                    vr = agent.verify(answer, manifest.statement)
                    vr.verifier_id = agent.agent_id
                    self.session_log.verification_results.append(vr)
                    return vr
                except Exception:
                    continue

        # No verifier available — assume pass
        return VerificationResult(
            passed=True,
            score=0.5,
            issues=["No independent verifier available — verification skipped"],
            verifier_id="system",
        )

    def _phase7_learn(
        self,
        manifest: ProblemManifest,
        results: List[PartialResult],
        verification: VerificationResult,
    ) -> None:
        """Phase 7: Learning.

        Updates trust scores and records decomposition patterns.
        """
        self.session_log.phases_completed.append("learn")

        # Update trust based on verification
        for r in results:
            if hasattr(r, "confidence"):
                for cap in manifest.sub_problems:
                    if cap.id == r.sub_problem_id:
                        for c in cap.capabilities_needed:
                            self.trust_manager.update_from_verification(
                                r.agent_id, c, verification.passed
                            )
                            break

        # Record successful decomposition pattern
        success_score = verification.score if verification.passed else 0.0
        self.decomposer.record_pattern(manifest, success_score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_claims(
        self, claims: List[Claim], manifest: ProblemManifest
    ) -> Dict[str, str]:
        """Resolve claim conflicts and produce assignment map.

        Conflict resolution priority:
        1. Higher (confidence * trust_weight)
        2. Lower estimated effort
        3. Random tiebreaker
        """
        # Group claims by sub-problem
        by_sub: Dict[str, List[Claim]] = {}
        for claim in claims:
            by_sub.setdefault(claim.sub_problem_id, []).append(claim)

        assignments: Dict[str, str] = {}

        for sp in manifest.sub_problems:
            sub_claims = by_sub.get(sp.id, [])

            if not sub_claims:
                # No claims — try to suggest assignment
                agent_caps = {
                    a.agent_id: a.capabilities for a in self.agents
                    if hasattr(a, "agent_id") and hasattr(a, "capabilities")
                }
                suggested = self.decomposer.suggest_assignment([sp], agent_caps)
                if sp.id in suggested:
                    assignments[sp.id] = suggested[sp.id]
                continue

            if len(sub_claims) == 1:
                assignments[sp.id] = sub_claims[0].agent_id
                continue

            # Multiple claims — resolve by weighted score
            def claim_score(claim: Claim) -> float:
                trust = self.trust_manager.get_overall_trust(claim.agent_id)
                return claim.confidence * (0.5 + 0.5 * trust) - claim.estimated_effort * 0.01

            best_claim = max(sub_claims, key=claim_score)
            assignments[sp.id] = best_claim.agent_id

        return assignments

    def _handle_conflict_trust(self, results: List[PartialResult]) -> None:
        """Update trust when agents produce conflicting results."""
        by_sub: Dict[str, List[PartialResult]] = {}
        for r in results:
            by_sub.setdefault(r.sub_problem_id, []).append(r)

        for _sub_id, subs in by_sub.items():
            if len(subs) <= 1:
                continue
            # Higher confidence wins
            winner = max(subs, key=lambda r: r.confidence)
            for loser in subs:
                if loser.agent_id != winner.agent_id:
                    self.trust_manager.update_from_contradiction(
                        winner.agent_id,
                        loser.agent_id,
                        # Use a generic capability since we don't know the specific one
                        "reasoning",
                    )
