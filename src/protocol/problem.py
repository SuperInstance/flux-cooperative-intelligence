"""
flux-cooperative-intelligence: Problem Decomposition Data Types

Defines the core data structures for cooperative problem-solving:
ProblemManifest, SubProblem, PartialResult, and the ProblemDecomposer engine.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SubProblemStatus(str, Enum):
    """Lifecycle states for a sub-problem."""
    UNCLAIMED = "unclaimed"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    YIELDED = "yielded"
    FAILED = "failed"


class DifficultyLevel(str, Enum):
    """Qualitative difficulty tiers (maps to 0.0–1.0 floats)."""
    TRIVIAL = "trivial"       # 0.0–0.2
    EASY = "easy"             # 0.2–0.4
    MODERATE = "moderate"     # 0.4–0.6
    HARD = "hard"             # 0.6–0.8
    EXPERT = "expert"         # 0.8–1.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SubProblem:
    """A single sub-problem within a larger cooperative effort.

    Attributes:
        id: Unique identifier for this sub-problem.
        description: What this sub-problem asks.
        capabilities_needed: Set of capability strings required to solve it.
        difficulty: Estimated difficulty as a float in [0.0, 1.0].
        dependencies: IDs of other sub-problems that must complete first.
        status: Current lifecycle status.
        assigned_agent: Agent ID if claimed/assigned, else None.
        result: PartialResult once solved, else None.
        parent_problem_id: ID of the ProblemManifest this belongs to.
    """
    id: str = field(default_factory=lambda: f"sp-{uuid.uuid4().hex[:8]}")
    description: str = ""
    capabilities_needed: Set[str] = field(default_factory=set)
    difficulty: float = 0.5
    dependencies: List[str] = field(default_factory=list)
    status: SubProblemStatus = SubProblemStatus.UNCLAIMED
    assigned_agent: Optional[str] = None
    result: Optional["PartialResult"] = None
    parent_problem_id: str = ""


@dataclass
class PartialResult:
    """The output of an agent solving a sub-problem.

    Attributes:
        sub_problem_id: Which sub-problem was addressed.
        agent_id: The agent that produced this result.
        answer: The actual answer (arbitrary type).
        confidence: Self-assessed correctness probability [0.0, 1.0].
        methodology: How the answer was derived.
        assumptions: Explicit assumptions made during solving.
        elapsed_time: Wall-clock seconds spent solving.
    """
    sub_problem_id: str = ""
    agent_id: str = ""
    answer: Any = None
    confidence: float = 0.5
    methodology: str = ""
    assumptions: List[str] = field(default_factory=list)
    elapsed_time: float = 0.0


@dataclass
class ProblemManifest:
    """Full specification of a problem decomposed for cooperative solving.

    Attributes:
        problem_id: Unique identifier.
        statement: Natural-language problem description.
        owner: Agent ID of the ProblemOwner.
        sub_problems: Ordered list of SubProblem instances.
        required_capabilities: Union of all capabilities across sub-problems.
        difficulty: Overall difficulty [0.0, 1.0] (max of sub-problems).
        created_at: ISO-8601 timestamp string.
        metadata: Arbitrary extra data.
    """
    problem_id: str = field(default_factory=lambda: f"prob-{uuid.uuid4().hex[:8]}")
    statement: str = ""
    owner: str = ""
    sub_problems: List[SubProblem] = field(default_factory=list)
    required_capabilities: Set[str] = field(default_factory=set)
    difficulty: float = 0.5
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive required_capabilities and difficulty from sub-problems."""
        if self.sub_problems:
            caps: Set[str] = set()
            max_diff = 0.0
            for sp in self.sub_problems:
                caps.update(sp.capabilities_needed)
                if sp.difficulty > max_diff:
                    max_diff = sp.difficulty
                sp.parent_problem_id = self.problem_id
            if not self.required_capabilities:
                self.required_capabilities = caps
            if self.difficulty == 0.5 and self.sub_problems:
                self.difficulty = max_diff


@dataclass
class Claim:
    """An agent's claim on a sub-problem during Phase 2.

    Attributes:
        sub_problem_id: Which sub-problem is being claimed.
        agent_id: Claiming agent.
        estimated_effort: Seconds the agent expects to spend.
        confidence: Self-assessed probability of solving correctly.
        approach: Brief description of the planned method.
        evidence: Supporting evidence (prior results, credentials, etc.).
    """
    sub_problem_id: str = ""
    agent_id: str = ""
    estimated_effort: float = 0.0
    confidence: float = 0.5
    approach: str = ""
    evidence: str = ""


@dataclass
class VerificationResult:
    """Output of Phase 6 verification.

    Attributes:
        passed: Whether verification succeeded.
        score: Quality score [0.0, 1.0].
        issues: Specific problems found.
        suggestions: Recommendations for improvement.
        verifier_id: Agent that performed verification.
    """
    passed: bool = False
    score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    verifier_id: str = ""


@dataclass
class CooperativeSolution:
    """The final output of a cooperative problem-solving session.

    Attributes:
        answer: The synthesized final answer.
        confidence: Overall confidence in the answer.
        methodology: How the answer was assembled (provenance narrative).
        agent_contributions: Map of agent_id -> list of sub_problem_ids they solved.
        total_time: Wall-clock seconds for the entire session.
        verification_result: Outcome of Phase 6, if run.
        problem_id: ID of the original problem.
    """
    answer: Any = None
    confidence: float = 0.0
    methodology: str = ""
    agent_contributions: Dict[str, List[str]] = field(default_factory=dict)
    total_time: float = 0.0
    verification_result: Optional[VerificationResult] = None
    problem_id: str = ""


# ---------------------------------------------------------------------------
# Problem Decomposer
# ---------------------------------------------------------------------------

class ProblemDecomposer:
    """Decomposes complex problem statements into structured ProblemManifests.

    The decomposer uses heuristic rules to:
    1. Split a problem into sub-problems by capability domain.
    2. Estimate difficulty for each sub-problem.
    3. Build a dependency graph between sub-problems.
    4. Suggest agent-to-sub-problem assignments based on capability profiles.

    This is a reference implementation — in production, an LLM-powered decomposer
    would be substituted for more intelligent decomposition.
    """

    # Capability keywords mapped to capability labels
    CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
        "math": ["calculate", "compute", "equation", "formula", "number",
                 "statistics", "probability", "optimize", "algorithm",
                 "complexity", "numeric"],
        "reasoning": ["analyze", "reason", "logic", "infer", "deduce",
                      "evaluate", "compare", "assess", "judge"],
        "coding": ["implement", "code", "function", "program", "debug",
                   "refactor", "test", "api", "database", "software"],
        "research": ["find", "search", "look up", "reference", "citation",
                     "data", "source", "literature", "document"],
        "writing": ["write", "draft", "compose", "summarize", "explain",
                    "document", "report", "narrative", "describe"],
        "design": ["design", "architect", "structure", "plan", "layout",
                   "schema", "model", "system"],
        "security": ["security", "vulnerability", "exploit", "encrypt",
                     "auth", "permission", "inject", "xss", "csrf"],
        "creativity": ["creative", "innovate", "brainstorm", "imagine",
                       "novel", "idea", "concept"],
    }

    def __init__(self) -> None:
        self._decomposition_templates: Dict[str, List[str]] = {}
        # Store learned decomposition patterns for reuse
        self._learned_patterns: List[Dict[str, Any]] = []

    def decompose(
        self,
        problem_statement: str,
        agent_capabilities: Optional[Dict[str, Set[str]]] = None,
    ) -> ProblemManifest:
        """Decompose a problem statement into a ProblemManifest.

        Args:
            problem_statement: Natural-language description of the problem.
            agent_capabilities: Optional map of agent_id -> capabilities,
                used to tailor decomposition to available skills.

        Returns:
            A ProblemManifest with sub-problems and metadata.
        """
        # Step 1: Identify required capabilities from the problem text
        detected_caps = self._detect_capabilities(problem_statement)

        # Step 2: Generate sub-problems (heuristic: one per major capability)
        sub_problems = self._generate_sub_problems(
            problem_statement, detected_caps
        )

        # Step 3: Build dependency graph
        self._identify_dependencies(sub_problems)

        # Step 4: Estimate difficulty
        for sp in sub_problems:
            sp.difficulty = self.estimate_difficulty(sp, problem_statement)

        # Step 5: Tailor if agent capabilities are known
        if agent_capabilities:
            self._tailor_to_agents(sub_problems, agent_capabilities)

        manifest = ProblemManifest(
            statement=problem_statement,
            sub_problems=sub_problems,
        )

        return manifest

    def estimate_difficulty(
        self,
        sub_problem: SubProblem,
        context: str = "",
    ) -> float:
        """Estimate difficulty of a sub-problem on a [0.0, 1.0] scale.

        Heuristics:
        - More capabilities needed -> higher difficulty
        - More dependencies -> higher difficulty
        - Longer description -> slightly higher (more complex)
        - Words suggesting complexity increase difficulty

        Args:
            sub_problem: The sub-problem to estimate.
            context: The original problem statement for additional signals.

        Returns:
            Difficulty float in [0.0, 1.0].
        """
        diff = 0.3  # base

        # Capability breadth
        cap_count = len(sub_problem.capabilities_needed)
        diff += min(cap_count * 0.1, 0.3)

        # Dependencies
        dep_count = len(sub_problem.dependencies)
        diff += min(dep_count * 0.05, 0.15)

        # Description complexity (very rough proxy)
        word_count = len(sub_problem.description.split())
        diff += min(word_count * 0.005, 0.1)

        # Complexity signals in text
        complexity_words = [
            "complex", "difficult", "intricate", "subtle", "nuanced",
            "ambiguous", "uncertain", "multi-step", "optimize",
        ]
        text = (sub_problem.description + " " + context).lower()
        for word in complexity_words:
            if word in text:
                diff += 0.05

        return min(max(diff, 0.0), 1.0)

    def identify_dependencies(
        self, sub_problems: List[SubProblem]
    ) -> Dict[str, List[str]]:
        """Build a dependency graph between sub-problems.

        Rules:
        - Sub-problems requiring "design" or "research" typically precede
          "coding" or "writing" sub-problems.
        - Later sub-problems in the list may depend on earlier ones if they
          share capabilities.
        - A sub-problem with broader capabilities may depend on more
          specialized sub-problems.
        - Cycles are detected and broken by removing the edge that would
          create the cycle.

        Args:
            sub_problems: List to analyze.

        Returns:
            Dict mapping sub_problem_id -> list of dependency IDs.
        """
        self._identify_dependencies(sub_problems)

        # Detect and break cycles
        self._detect_and_break_cycles(sub_problems)

        graph: Dict[str, List[str]] = {}
        for sp in sub_problems:
            graph[sp.id] = list(sp.dependencies)

        return graph

    def has_cycle(self, sub_problems: List[SubProblem]) -> bool:
        """Check whether the dependency graph contains a cycle.

        Args:
            sub_problems: List of sub-problems to check.

        Returns:
            True if a cycle exists, False otherwise.
        """
        graph: Dict[str, List[str]] = {}
        for sp in sub_problems:
            graph[sp.id] = list(sp.dependencies)

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for sp in sub_problems:
            if sp.id not in visited:
                if dfs(sp.id):
                    return True
        return False

    def suggest_assignment(
        self,
        sub_problems: List[SubProblem],
        agent_capabilities: Dict[str, Set[str]],
    ) -> Dict[str, str]:
        """Suggest agent assignments for sub-problems.

        Uses a greedy matching approach: for each sub-problem, find the agent
        with the best capability coverage and assign them. Agents can be
        assigned multiple sub-problems.

        Args:
            sub_problems: Sub-problems needing assignment.
            agent_capabilities: Map of agent_id -> capability set.

        Returns:
            Dict mapping sub_problem_id -> agent_id.
        """
        assignments: Dict[str, str] = {}
        agent_load: Dict[str, int] = {a: 0 for a in agent_capabilities}

        for sp in sub_problems:
            best_agent = None
            best_score = -1.0

            for agent_id, caps in agent_capabilities.items():
                # Capability overlap score
                overlap = len(sp.capabilities_needed & caps)
                coverage = overlap / max(len(sp.capabilities_needed), 1)

                # Penalize agents already heavily loaded (load balancing)
                load_penalty = agent_load.get(agent_id, 0) * 0.1
                score = coverage - load_penalty

                if score > best_score:
                    best_score = score
                    best_agent = agent_id

            if best_agent and best_score > 0:
                assignments[sp.id] = best_agent
                agent_load[best_agent] = agent_load.get(best_agent, 0) + 1

        return assignments

    def record_pattern(
        self, manifest: ProblemManifest, success_score: float
    ) -> None:
        """Record a decomposition pattern for future reuse.

        Args:
            manifest: The problem manifest that was used.
            success_score: How well this decomposition worked (0.0–1.0).
        """
        pattern = {
            "capabilities": list(manifest.required_capabilities),
            "sub_problem_count": len(manifest.sub_problems),
            "difficulty": manifest.difficulty,
            "success_score": success_score,
            "statement_hash": hash(manifest.statement) % (10**8),
        }
        self._learned_patterns.append(pattern)

    def get_similar_patterns(
        self, capabilities: Set[str], difficulty: float
    ) -> List[Dict[str, Any]]:
        """Retrieve previously successful patterns similar to the query.

        Args:
            capabilities: Required capabilities to match.
            difficulty: Difficulty level to match.

        Returns:
            List of pattern dicts, sorted by relevance.
        """
        scored = []
        for pat in self._learned_patterns:
            pat_caps = set(pat["capabilities"])
            cap_overlap = len(pat_caps & capabilities) / max(len(capabilities), 1)
            diff_distance = abs(pat["difficulty"] - difficulty)
            relevance = cap_overlap * (1.0 - diff_distance) * pat["success_score"]
            scored.append((relevance, pat))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pat for _, pat in scored]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_capabilities(self, text: str) -> List[str]:
        """Detect required capabilities from problem text."""
        text_lower = text.lower()
        detected: List[str] = []
        scores: Dict[str, int] = {}

        for cap, keywords in self.CAPABILITY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[cap] = count

        # Sort by score descending, take top capabilities
        for cap, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            detected.append(cap)

        # If nothing detected, default to reasoning
        if not detected:
            detected = ["reasoning"]

        return detected

    def _generate_sub_problems(
        self,
        problem_statement: str,
        capabilities: List[str],
    ) -> List[SubProblem]:
        """Generate sub-problems from detected capabilities."""
        sub_problems: List[SubProblem] = []

        # For each detected capability, create a focused sub-problem
        for i, cap in enumerate(capabilities):
            desc = self._generate_sub_problem_description(
                problem_statement, cap, i + 1, len(capabilities)
            )
            sp = SubProblem(
                description=desc,
                capabilities_needed={cap},
                difficulty=0.5,  # Will be refined by estimate_difficulty
            )
            sub_problems.append(sp)

        # If only one capability detected but the problem seems complex,
        # try to create 2 sub-problems for parallel approaches
        if len(sub_problems) == 1:
            word_count = len(problem_statement.split())
            if word_count > 30:
                cap = capabilities[0]
                original = sub_problems[0]
                # Split into "analyze" and "solve"
                sp1 = SubProblem(
                    description=(
                        f"Analysis phase: Break down the core aspects of: "
                        f"{problem_statement[:200]}"
                    ),
                    capabilities_needed={cap},
                    difficulty=0.4,
                )
                sp2 = SubProblem(
                    description=(
                        f"Solution phase: Propose solutions based on analysis of: "
                        f"{problem_statement[:200]}"
                    ),
                    capabilities_needed={cap},
                    difficulty=0.6,
                    dependencies=[sp1.id],
                )
                sub_problems = [sp1, sp2]
            else:
                # Simple problem, just one sub-problem
                sub_problems[0].description = (
                    f"Solve: {problem_statement}"
                )

        return sub_problems

    def _generate_sub_problem_description(
        self,
        problem_statement: str,
        capability: str,
        index: int,
        total: int,
    ) -> str:
        """Generate a focused sub-problem description for a capability."""
        templates = {
            "math": (
                f"Perform mathematical analysis and computation for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "reasoning": (
                f"Apply logical reasoning and analysis to "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "coding": (
                f"Implement and test code solutions for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "research": (
                f"Research and gather information for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "writing": (
                f"Draft and refine written content for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "design": (
                f"Design and structure the architecture for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "security": (
                f"Perform security analysis for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
            "creativity": (
                f"Generate creative approaches and ideas for "
                f"aspect {index}/{total} of: {problem_statement[:200]}"
            ),
        }
        return templates.get(capability, (
            f"Apply {capability} expertise to "
            f"aspect {index}/{total} of: {problem_statement[:200]}"
        ))

    def _identify_dependencies(
        self, sub_problems: List[SubProblem]
    ) -> None:
        """Mutate sub-problems to add dependency edges."""
        # Ordering heuristic: research/design come before coding/writing
        precedence = {
            "research": 0,
            "design": 1,
            "reasoning": 1,
            "creativity": 2,
            "math": 2,
            "security": 3,
            "coding": 4,
            "writing": 4,
        }

        for i, sp in enumerate(sub_problems):
            for j, other in enumerate(sub_problems):
                if i == j:
                    continue
                # Get primary capability of each
                sp_cap = next(iter(sp.capabilities_needed), "") if sp.capabilities_needed else ""
                other_cap = next(iter(other.capabilities_needed), "") if other.capabilities_needed else ""

                sp_order = precedence.get(sp_cap, 2)
                other_order = precedence.get(other_cap, 2)

                # If other should come before this one, add dependency
                if other_order < sp_order and j < i:
                    if other.id not in sp.dependencies:
                        sp.dependencies.append(other.id)

    def _detect_and_break_cycles(
        self, sub_problems: List[SubProblem]
    ) -> None:
        """Detect cycles in the dependency graph and break them.

        Uses iterative DFS to find back-edges and removes the dependency
        that would close the cycle (the last edge in the cycle).
        """
        id_to_sp: Dict[str, SubProblem] = {sp.id: sp for sp in sub_problems}

        max_iterations = len(sub_problems) + 1  # Safety bound
        for _ in range(max_iterations):
            # Build adjacency list
            graph: Dict[str, List[str]] = {}
            for sp in sub_problems:
                graph[sp.id] = list(sp.dependencies)

            # DFS cycle detection
            visited: Set[str] = set()
            rec_stack: Set[str] = set()
            cycle_edge: Optional[Tuple[str, str]] = None

            def _dfs(node: str) -> bool:
                nonlocal cycle_edge
                visited.add(node)
                rec_stack.add(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in id_to_sp:
                        continue  # Skip dangling references
                    if neighbor not in visited:
                        if _dfs(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        cycle_edge = (node, neighbor)
                        return True
                rec_stack.discard(node)
                return False

            has_cycle = False
            for sp in sub_problems:
                if sp.id not in visited:
                    if _dfs(sp.id):
                        has_cycle = True
                        break

            if not has_cycle or cycle_edge is None:
                break

            # Break the cycle by removing the edge (from -> to)
            from_id, to_id = cycle_edge
            if from_id in id_to_sp:
                sp = id_to_sp[from_id]
                if to_id in sp.dependencies:
                    sp.dependencies.remove(to_id)

    def _tailor_to_agents(
        self,
        sub_problems: List[SubProblem],
        agent_capabilities: Dict[str, Set[str]],
    ) -> None:
        """Adjust sub-problem decomposition based on available agents.

        If no agent has a required capability, merge sub-problems or
        add fallback capabilities.
        """
        all_caps: Set[str] = set()
        for caps in agent_capabilities.values():
            all_caps.update(caps)

        for sp in sub_problems:
            missing = sp.capabilities_needed - all_caps
            if missing and all_caps:
                # Replace missing caps with the closest available cap
                sp.capabilities_needed -= missing
                # Add the most broadly available capability as fallback
                sp.capabilities_needed.add(next(iter(all_caps)))
