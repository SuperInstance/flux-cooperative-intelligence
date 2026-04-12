"""flux-cooperative-intelligence: Cooperative Intelligence Compiler.

Translates high-level cooperative intelligence programs into FLUX bytecode
using the A2A opcode set (TELL, ASK, DELEG, BCAST, ACCEPT, DECLINE, REPORT,
MERGE, FORK, JOIN, SIGNAL, AWAIT, TRUST, DISCOV, STATUS, HEARTBT).
"""

from .compiler import (
    # Opcodes
    A2A_Opcodes,
    SupportOpcodes,
    # IR nodes
    Task,
    Divide,
    Delegate,
    Broadcast,
    Ask,
    Tell,
    Merge,
    MergeStrategy,
    Fork,
    Join,
    Trust,
    Report,
    Signal,
    Await,
    CoopProgram,
    # Code generator
    CodeGenerator,
    FluxInstruction,
    # Optimizer
    Optimizer,
    # Pattern library
    PatternLibrary,
    # High-level compiler interface
    compile_coop_program,
)

__all__ = [
    # Opcodes
    "A2A_Opcodes",
    "SupportOpcodes",
    # IR nodes
    "Task",
    "Divide",
    "Delegate",
    "Broadcast",
    "Ask",
    "Tell",
    "Merge",
    "MergeStrategy",
    "Fork",
    "Join",
    "Trust",
    "Report",
    "Signal",
    "Await",
    "CoopProgram",
    # Code generator
    "CodeGenerator",
    "FluxInstruction",
    # Optimizer
    "Optimizer",
    # Pattern library
    "PatternLibrary",
    # High-level compiler interface
    "compile_coop_program",
]
