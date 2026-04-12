"""
flux-cooperative-intelligence: Alternative Collaboration Patterns

Three patterns for different cooperative problem-solving scenarios:

1. MapReducePattern — Embarrassingly parallel problems.
2. DebatePattern — Problems where agents disagree and must argue.
3. CascadePattern — Sequential processing with feedback.

Each pattern is a self-contained strategy that can be used instead of (or
alongside) the full DCS protocol depending on problem characteristics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Map-Reduce Pattern
# ---------------------------------------------------------------------------

@dataclass
class MapReduceResult:
    """Output of a MapReduce cooperative session."""
    map_results: List[Any] = field(default_factory=list)
    reduced_answer: Any = None
    total_time: float = 0.0
    items_processed: int = 0
    agents_used: int = 0


class MapReducePattern:
    """Pattern for embarrassingly parallel problems.

    The same operation is applied independently to many data items,
    then results are combined with a reduction function.

    When to use:
    - Batch processing of independent items
    - Each item requires the same type of analysis
    - No dependencies between items

    Example:
        Analyze 100 code files for security vulnerabilities.
        Classify 500 documents by topic.
        Generate unit tests for 20 functions.
    """

    def __init__(self, agents: List[Any]) -> None:
        self.agents = agents

    def run(
        self,
        data_items: List[Any],
        map_fn: Callable[[Any, Any], Any],
        reduce_fn: Callable[[List[Any]], Any],
    ) -> MapReduceResult:
        """Execute the MapReduce pattern.

        Args:
            data_items: List of independent data items to process.
            map_fn: Function(agent, item) -> result. Each agent applies this.
            reduce_fn: Function(list_of_results) -> final_answer.
                Combines all partial results.

        Returns:
            MapReduceResult with final answer and metadata.
        """
        start = time.time()

        # Distribute items across agents (round-robin)
        map_results: List[Any] = []
        agents_used: Set[str] = set()

        for i, item in enumerate(data_items):
            agent = self.agents[i % len(self.agents)]
            if hasattr(agent, "agent_id"):
                agents_used.add(agent.agent_id)

            try:
                result = map_fn(agent, item)
                map_results.append(result)
            except Exception as e:
                map_results.append({"error": str(e), "item": item})

        # Reduce all results
        reduced = reduce_fn(map_results)

        return MapReduceResult(
            map_results=map_results,
            reduced_answer=reduced,
            total_time=time.time() - start,
            items_processed=len(data_items),
            agents_used=len(agents_used),
        )


# ---------------------------------------------------------------------------
# Debate Pattern
# ---------------------------------------------------------------------------

@dataclass
class DebateProposal:
    """A proposal in a debate round."""
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_id: str = ""
    answer: Any = None
    evidence: str = ""
    confidence: float = 0.5
    round_number: int = 0


@dataclass
class DebateArgument:
    """An argument for or against a proposal."""
    argument_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_id: str = ""
    for_proposal_id: str = ""
    position: str = "for"  # "for" or "against"
    argument: str = ""
    evidence: str = ""
    round_number: int = 0


@dataclass
class DebateVote:
    """A vote cast during the debate."""
    agent_id: str = ""
    for_proposal_id: str = ""
    weight: float = 1.0
    reason: str = ""


@dataclass
class DebateResult:
    """Output of a debate cooperative session."""
    winning_answer: Any = None
    winning_proposal_id: str = ""
    vote_distribution: Dict[str, float] = field(default_factory=dict)
    proposals: List[DebateProposal] = field(default_factory=list)
    arguments: List[DebateArgument] = field(default_factory=list)
    votes: List[DebateVote] = field(default_factory=list)
    dissent_register: List[Dict[str, Any]] = field(default_factory=list)
    rounds_completed: int = 0
    consensus_reached: bool = False
    total_time: float = 0.0


class DebatePattern:
    """Pattern for problems where agents disagree and need to argue.

    Structured debate with proposal, argument, voting, and resolution phases.

    When to use:
    - Design decisions with trade-offs
    - Diagnosing ambiguous issues
    - Problems where reasonable agents might legitimately disagree

    Termination:
    - All agents agree (consensus)
    - Maximum rounds reached
    - Vote distribution stabilizes (convergence)
    """

    MAX_ROUNDS = 5
    CONVERGENCE_THRESHOLD = 0.05  # Vote share change needed to continue

    def __init__(
        self,
        agents: List[Any],
        trust_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.agents = agents
        self.trust_weights = trust_weights or {}

    def run(
        self,
        problem_statement: str,
        propose_fn: Callable[[Any, str], Optional[DebateProposal]],
        argue_fn: Optional[Callable[[Any, List[DebateProposal]], Optional[DebateArgument]]] = None,
        vote_fn: Optional[Callable[[Any, List[DebateProposal]], Optional[DebateVote]]] = None,
    ) -> DebateResult:
        """Execute the Debate pattern.

        Args:
            problem_statement: The problem being debated.
            propose_fn: Function(agent, problem) -> DebateProposal or None.
                Each agent proposes their answer.
            argue_fn: Optional function(agent, proposals) -> DebateArgument or None.
                Agents argue for/against proposals. If None, skips argument phase.
            vote_fn: Optional function(agent, proposals) -> DebateVote or None.
                Agents vote on proposals. If None, uses confidence-weighted
                automatic voting.

        Returns:
            DebateResult with winning answer and full debate history.
        """
        start = time.time()
        result = DebateResult()

        # Step 1: PROPOSE — Each agent proposes their answer
        for agent in self.agents:
            proposal = propose_fn(agent, problem_statement)
            if proposal is not None:
                proposal.round_number = 1
                result.proposals.append(proposal)

        if not result.proposals:
            return result

        # If all agents agree, we're done
        unique_answers = set(str(p.answer) for p in result.proposals)
        if len(unique_answers) == 1:
            result.consensus_reached = True
            result.winning_answer = result.proposals[0].answer
            result.winning_proposal_id = result.proposals[0].proposal_id
            result.total_time = time.time() - start
            return result

        # Step 2 & 3: ARGUE and VOTE — Iterate for multiple rounds
        prev_distribution: Dict[str, float] = {}

        for round_num in range(1, self.MAX_ROUNDS + 1):
            # ARGUE phase
            if argue_fn is not None:
                for agent in self.agents:
                    argument = argue_fn(agent, result.proposals)
                    if argument is not None:
                        argument.round_number = round_num
                        result.arguments.append(argument)

            # VOTE phase
            round_votes: List[DebateVote] = []
            for agent in self.agents:
                agent_id = getattr(agent, "agent_id", str(id(agent)))
                weight = self.trust_weights.get(agent_id, 1.0)

                if vote_fn is not None:
                    vote = vote_fn(agent, result.proposals)
                    if vote is not None:
                        vote.weight = weight
                        round_votes.append(vote)
                else:
                    # Automatic voting: agent votes for its own proposal
                    my_proposal = next(
                        (p for p in result.proposals if p.agent_id == agent_id),
                        result.proposals[0],  # fallback
                    )
                    round_votes.append(DebateVote(
                        agent_id=agent_id,
                        for_proposal_id=my_proposal.proposal_id,
                        weight=weight,
                        reason="Own proposal",
                    ))

            result.votes.extend(round_votes)

            # Tally votes
            distribution = self._tally_votes(round_votes)

            # Check convergence
            if prev_distribution and self._is_converged(prev_distribution, distribution):
                break

            prev_distribution = distribution

            # Check consensus
            total_weight = sum(distribution.values())
            if total_weight > 0:
                max_share = max(distribution.values()) / total_weight
                if max_share > 0.9:  # 90%+ agreement
                    result.consensus_reached = True
                    break

            result.rounds_completed = round_num

        # Step 4: RESOLVE — Majority wins
        result.vote_distribution = distribution
        if distribution:
            winner_id = max(distribution, key=distribution.get)
            winner_proposal = next(
                (p for p in result.proposals if p.proposal_id == winner_id),
                None,
            )
            if winner_proposal:
                result.winning_answer = winner_proposal.answer
                result.winning_proposal_id = winner_proposal.proposal_id

            # Build dissent register
            for prop in result.proposals:
                if prop.proposal_id != winner_id:
                    share = distribution.get(prop.proposal_id, 0)
                    if share > 0:
                        result.dissent_register.append({
                            "proposal_id": prop.proposal_id,
                            "agent_id": prop.agent_id,
                            "answer": prop.answer,
                            "vote_share": share,
                            "evidence": prop.evidence,
                        })

        result.total_time = time.time() - start
        return result

    def _tally_votes(self, votes: List[DebateVote]) -> Dict[str, float]:
        """Tally votes weighted by agent trust."""
        distribution: Dict[str, float] = {}
        for vote in votes:
            distribution[vote.for_proposal_id] = (
                distribution.get(vote.for_proposal_id, 0.0) + vote.weight
            )
        return distribution

    def _is_converged(
        self, prev: Dict[str, float], curr: Dict[str, float]
    ) -> bool:
        """Check if vote distribution has stabilized."""
        all_ids = set(prev.keys()) | set(curr.keys())
        total_change = 0.0
        total_weight = sum(curr.values()) or 1.0

        for pid in all_ids:
            p = prev.get(pid, 0.0) / total_weight
            c = curr.get(pid, 0.0) / total_weight
            total_change += abs(c - p)

        return total_change < self.CONVERGENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Cascade Pattern
# ---------------------------------------------------------------------------

@dataclass
class CascadeStep:
    """A single step in a cascade pipeline."""
    step_number: int = 0
    agent_id: str = ""
    input_data: Any = None
    output_data: Any = None
    feedback_requests: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_time: float = 0.0
    status: str = "pending"  # pending, completed, feedback_requested


@dataclass
class CascadeResult:
    """Output of a Cascade cooperative session."""
    final_answer: Any = None
    steps: List[CascadeStep] = field(default_factory=list)
    total_time: float = 0.0
    feedback_loops: int = 0
    pipeline_completed: bool = False


class CascadePattern:
    """Pattern for sequential processing where each agent builds on
    the previous agent's output.

    Pipeline: Agent A -> Agent B -> Agent C -> ... -> Final

    Feedback mechanism: Any agent can request revisions from an earlier
    agent by issuing a YIELD-BACK with specific feedback.

    When to use:
    - Pipeline processing (e.g., outline -> draft -> edit -> proofread)
    - Problems with natural sequential dependencies
    - When each stage requires different expertise
    """

    MAX_FEEDBACK_LOOPS = 3

    def __init__(self, agent_chain: List[Any]) -> None:
        """Initialize with an ordered list of agents forming the pipeline.

        Args:
            agent_chain: Agents in processing order. Agent 0 receives the
                initial input; the last agent produces the final output.
        """
        self.agent_chain = agent_chain

    def run(
        self,
        initial_input: Any,
        process_fn: Callable[[Any, Any], Tuple[Any, Optional[Dict[str, Any]]]],
    ) -> CascadeResult:
        """Execute the Cascade pattern.

        Args:
            initial_input: The starting data for the pipeline.
            process_fn: Function(agent, input_data) -> (output, feedback_request).
                The feedback_request is optional; if provided, it's a dict with
                keys: {"step": int, "feedback": str, "input": Any} indicating
                which earlier step should be revised.

        Returns:
            CascadeResult with final answer and pipeline history.
        """
        start = time.time()
        result = CascadeResult()
        current_data = initial_input
        feedback_loops = 0

        # Run the pipeline (with potential feedback loops)
        max_iterations = (len(self.agent_chain) + 1) * (self.MAX_FEEDBACK_LOOPS + 1)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            pipeline_completed = True

            for i, agent in enumerate(self.agent_chain):
                step = CascadeStep(
                    step_number=i,
                    agent_id=getattr(agent, "agent_id", f"agent-{i}"),
                    input_data=current_data,
                    status="in_progress",
                )

                step_start = time.time()
                try:
                    output, feedback = process_fn(agent, current_data)
                    step.output_data = output
                    step.elapsed_time = time.time() - step_start
                    step.status = "completed"
                    current_data = output

                    # Handle feedback request (YIELD-BACK)
                    if feedback is not None:
                        target_step = feedback.get("step", i - 1)
                        if 0 <= target_step < i:
                            step.feedback_requests.append(feedback)
                            result.steps.append(step)
                            # Rewind to target step
                            target = result.steps[target_step]
                            current_data = target.input_data
                            feedback_loops += 1
                            pipeline_completed = False
                            break  # Restart pipeline from target step
                except Exception as e:
                    step.status = "failed"
                    step.output_data = {"error": str(e)}
                    step.elapsed_time = time.time() - step_start
                    result.steps.append(step)
                    result.total_time = time.time() - start
                    result.feedback_loops = feedback_loops
                    return result

                result.steps.append(step)

            if pipeline_completed:
                break

        result.final_answer = current_data
        result.pipeline_completed = pipeline_completed
        result.total_time = time.time() - start
        result.feedback_loops = feedback_loops
        return result


# ---------------------------------------------------------------------------
# Pattern Selector
# ---------------------------------------------------------------------------

@dataclass
class PatternRecommendation:
    """Recommendation for which pattern to use."""
    pattern_name: str
    confidence: float
    reasoning: str


class PatternSelector:
    """Recommends the best collaboration pattern for a given problem.

    Heuristics:
    - MapReduce: Large batch of independent items, same operation.
    - Debate: Multiple valid approaches, design decisions, ambiguity.
    - Cascade: Natural sequential pipeline, each step different expertise.
    - DCS: Complex multi-faceted problem (default).
    """

    def recommend(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PatternRecommendation:
        """Analyze a problem and recommend a collaboration pattern.

        Args:
            problem_statement: The problem description.
            context: Optional metadata (e.g., number of items, agent types).

        Returns:
            PatternRecommendation with pattern name and reasoning.
        """
        text = problem_statement.lower()
        scores: Dict[str, float] = {}

        # MapReduce signals
        map_reduce_signals = [
            "each", "every", "all", "batch", "list of", "items",
            "files", "documents", "independently", "parallel",
        ]
        mr_score = sum(1 for s in map_reduce_signals if s in text)
        if context and context.get("item_count", 0) > 5:
            mr_score += 2
        scores["map_reduce"] = min(mr_score * 0.15, 1.0)

        # Debate signals
        debate_signals = [
            "decide", "choose", "which", "compare", "trade-off",
            "opinion", "disagree", "pros and cons", "alternative",
            "best approach", "recommend", "versus", "vs",
        ]
        debate_score = sum(1 for s in debate_signals if s in text)
        scores["debate"] = min(debate_score * 0.15, 1.0)

        # Cascade signals
        cascade_signals = [
            "then", "followed by", "pipeline", "sequence", "first",
            "after that", "refine", "review", "edit", "proofread",
            "draft", "outline", "iterate",
        ]
        cascade_score = sum(1 for s in cascade_signals if s in text)
        scores["cascade"] = min(cascade_score * 0.15, 1.0)

        # DCS (default for complex multi-faceted problems)
        dcs_signals = [
            "complex", "multiple aspects", "several parts",
            "requires", "analyze and", "design and implement",
            "research and",
        ]
        dcs_score = sum(1 for s in dcs_signals if s in text) + 0.3  # base score
        scores["dcs"] = min(dcs_score * 0.15, 1.0)

        # Pick the highest-scoring pattern
        best_pattern = max(scores, key=scores.get)
        best_score = scores[best_pattern]

        reasoning_map = {
            "map_reduce": "Problem appears to involve batch processing of independent items — MapReduce is recommended.",
            "debate": "Problem involves competing alternatives or design decisions — Debate is recommended.",
            "cascade": "Problem has natural sequential dependencies — Cascade is recommended.",
            "dcs": "Problem is complex and multi-faceted — full DCS protocol is recommended.",
        }

        return PatternRecommendation(
            pattern_name=best_pattern,
            confidence=best_score,
            reasoning=reasoning_map[best_pattern],
        )
