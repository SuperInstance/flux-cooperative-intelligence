"""flux-cooperative-intelligence: Protocol sub-package."""
from .problem import (
    Claim,
    CooperativeSolution,
    DifficultyLevel,
    PartialResult,
    ProblemDecomposer,
    ProblemManifest,
    SubProblem,
    SubProblemStatus,
    VerificationResult,
)
from .executor import (
    AgentInterface,
    CommCallbacks,
    DCSExecutor,
    SessionLog,
    Synthesizer,
    TrustManager,
)
from .patterns import (
    CascadePattern,
    CascadeResult,
    CascadeStep,
    DebateArgument,
    DebatePattern,
    DebateProposal,
    DebateResult,
    DebateVote,
    MapReducePattern,
    MapReduceResult,
    PatternRecommendation,
    PatternSelector,
)

__all__ = [
    # Problem types
    "Claim",
    "CooperativeSolution",
    "DifficultyLevel",
    "PartialResult",
    "ProblemDecomposer",
    "ProblemManifest",
    "SubProblem",
    "SubProblemStatus",
    "VerificationResult",
    # Executor
    "AgentInterface",
    "CommCallbacks",
    "DCSExecutor",
    "SessionLog",
    "Synthesizer",
    "TrustManager",
    # Patterns
    "CascadePattern",
    "CascadeResult",
    "CascadeStep",
    "DebateArgument",
    "DebatePattern",
    "DebateProposal",
    "DebateResult",
    "DebateVote",
    "MapReducePattern",
    "MapReduceResult",
    "PatternRecommendation",
    "PatternSelector",
]
