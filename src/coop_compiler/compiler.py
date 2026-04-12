"""
Cooperative Intelligence Compiler — IR, Codegen, Patterns, Optimizer.

Translates high-level cooperative intelligence programs (Coop IR) into FLUX
bytecode using the A2A opcode set.

Register allocation:
    R0  = scratch register
    R1-R7   = task data registers
    R8-R15  = agent handle registers

A2A opcodes:
    TELL    = 0x50   ASK     = 0x51   DELEG   = 0x52
    BCAST   = 0x53   ACCEPT  = 0x54   DECLINE = 0x55
    REPORT  = 0x56   MERGE   = 0x57   FORK    = 0x58
    JOIN    = 0x59   SIGNAL  = 0x5A   AWAIT   = 0x5B
    TRUST   = 0x5C   DISCOV  = 0x5D   STATUS  = 0x5E
    HEARTBT = 0x5F

Supporting opcodes:
    HALT    = 0x00   NOP     = 0x01
    MOV     = 0x3A   MOVI16  = 0x40
    ADD     = 0x20   CMP_EQ  = 0x2C   JNZ     = 0x3D
    PUSH    = 0x0C   POP     = 0x0D
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# =========================================================================
# Opcode definitions
# =========================================================================

class A2A_Opcodes(enum.IntEnum):
    """FLUX A2A (Agent-to-Agent) opcodes for cooperative intelligence."""

    TELL = 0x50
    ASK = 0x51
    DELEG = 0x52
    BCAST = 0x53
    ACCEPT = 0x54
    DECLINE = 0x55
    REPORT = 0x56
    MERGE = 0x57
    FORK = 0x58
    JOIN = 0x59
    SIGNAL = 0x5A
    AWAIT = 0x5B
    TRUST = 0x5C
    DISCOV = 0x5D
    STATUS = 0x5E
    HEARTBT = 0x5F


class SupportOpcodes(enum.IntEnum):
    """FLUX supporting opcodes used by the cooperative compiler."""

    HALT = 0x00
    NOP = 0x01
    ADD = 0x20
    CMP_EQ = 0x2C
    PUSH = 0x0C
    POP = 0x0D
    MOV = 0x3A
    MOVI16 = 0x40
    JNZ = 0x3D


# =========================================================================
# Merge strategies
# =========================================================================

class MergeStrategy(enum.Enum):
    """Strategies for combining partial results."""

    SUM = "SUM"
    MAX = "MAX"
    MIN = "MIN"
    VOTE = "VOTE"
    CONCAT = "CONCAT"


# =========================================================================
# Coop IR — Intermediate Representation Nodes
# =========================================================================

@dataclass
class Task:
    """A unit of work in a cooperative program.

    Attributes:
        name: Human-readable task identifier.
        agent: Agent identifier (name, ID, or register).
        payload: Task payload data (arbitrary).
        priority: Task priority (higher = more urgent).
    """
    name: str
    agent: str
    payload: Any = None
    priority: int = 0


@dataclass
class Divide:
    """Split a task into subtasks for parallel processing.

    Attributes:
        task: The parent task to divide.
        subtasks: List of child subtasks.
    """
    task: Task
    subtasks: List[Task] = field(default_factory=list)


@dataclass
class Delegate:
    """Send a task to a specific agent.

    Attributes:
        task: The task to delegate.
        target_agent: Destination agent identifier.
    """
    task: Task
    target_agent: str


@dataclass
class Broadcast:
    """Broadcast a task to a fleet subset.

    Attributes:
        task: The task to broadcast.
        fleet_mask: Bitmask or list of agent identifiers for the target subset.
    """
    task: Task
    fleet_mask: Union[int, List[str]] = 0


@dataclass
class Ask:
    """Request data from an agent.

    Attributes:
        agent: Target agent identifier.
        query: The query string or data.
    """
    agent: str
    query: Any


@dataclass
class Tell:
    """Push data to an agent.

    Attributes:
        agent: Target agent identifier.
        data: The data to send.
    """
    agent: str
    data: Any


@dataclass
class Merge:
    """Combine partial results using a strategy.

    Attributes:
        results: List of result identifiers or values.
        strategy: The merge strategy to apply.
    """
    results: List[Any]
    strategy: MergeStrategy = MergeStrategy.SUM


@dataclass
class Fork:
    """Spawn parallel agents/branches.

    Attributes:
        state: The state to fork from.
        count: Number of parallel branches to create.
    """
    state: Any = None
    count: int = 2


@dataclass
class Join:
    """Wait for all children to complete.

    Attributes:
        children: List of child identifiers or branches to join.
    """
    children: List[Any] = field(default_factory=list)


@dataclass
class Trust:
    """Set trust level for an agent.

    Attributes:
        agent: Target agent identifier.
        level: Trust level (0.0 to 1.0).
    """
    agent: str
    level: float = 0.5


@dataclass
class Report:
    """Report progress or status.

    Attributes:
        status: Status string or code.
        data: Additional data to include in the report.
    """
    status: str
    data: Any = None


@dataclass
class Signal:
    """Emit a named signal.

    Attributes:
        name: Signal name.
        data: Data payload attached to the signal.
    """
    name: str
    data: Any = None


@dataclass
class Await:
    """Wait for a named signal.

    Attributes:
        signal_name: Name of the signal to wait for.
    """
    signal_name: str


# Union type for all IR nodes
CoopIRNode = Union[
    Task, Divide, Delegate, Broadcast, Ask, Tell, Merge,
    Fork, Join, Trust, Report, Signal, Await,
]


@dataclass
class CoopProgram:
    """A cooperative intelligence program — a sequence of IR nodes.

    Attributes:
        name: Program name.
        nodes: Ordered list of IR nodes (the program body).
    """
    name: str
    nodes: List[CoopIRNode] = field(default_factory=list)

    def add(self, node: CoopIRNode) -> "CoopProgram":
        """Fluent API: append a node and return self."""
        self.nodes.append(node)
        return self


# =========================================================================
# FLUX Instruction — the output of code generation
# =========================================================================

@dataclass
class FluxInstruction:
    """A single FLUX bytecode instruction.

    Attributes:
        opcode: The instruction opcode (int).
        operands: List of operand values (registers, immediates, labels).
        comment: Optional human-readable comment.
        label: Optional label name (for branch targets).
    """
    opcode: int
    operands: List[int] = field(default_factory=list)
    comment: str = ""
    label: str = ""

    def __repr__(self) -> str:
        ops = " ".join(str(o) for o in self.operands)
        prefix = f"{self.label}: " if self.label else ""
        comment = f"  ; {self.comment}" if self.comment else ""
        return f"{prefix}0x{self.opcode:02X} {ops}{comment}"

    def to_bytes(self) -> bytes:
        """Serialize to raw bytecode bytes."""
        result = bytes([self.opcode])
        for op in self.operands:
            if 0 <= op <= 0xFF:
                result += bytes([op])
            elif 0x100 <= op <= 0xFFFF:
                result += op.to_bytes(2, "big", signed=False)
            else:
                result += op.to_bytes(4, "big", signed=False)
        return result

    @staticmethod
    def halt(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.HALT, comment=comment or "halt")

    @staticmethod
    def nop(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.NOP, comment=comment or "nop")

    @staticmethod
    def mov(dst: int, src: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.MOV, [dst, src],
                               comment=comment or f"mov R{dst}, R{src}")

    @staticmethod
    def movi16(dst: int, imm: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.MOVI16, [dst, imm],
                               comment=comment or f"movi16 R{dst}, {imm}")

    @staticmethod
    def add(dst: int, lhs: int, rhs: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.ADD, [dst, lhs, rhs],
                               comment=comment or f"add R{dst}, R{lhs}, R{rhs}")

    @staticmethod
    def cmp_eq(a: int, b: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.CMP_EQ, [a, b],
                               comment=comment or f"cmp_eq R{a}, R{b}")

    @staticmethod
    def jnz(label_or_offset: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.JNZ, [label_or_offset],
                               comment=comment or f"jnz {label_or_offset}")

    @staticmethod
    def push(reg: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.PUSH, [reg],
                               comment=comment or f"push R{reg}")

    @staticmethod
    def pop(reg: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(SupportOpcodes.POP, [reg],
                               comment=comment or f"pop R{reg}")

    @staticmethod
    def tell(agent_reg: int, data_reg: int,
             comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.TELL, [agent_reg, data_reg],
                               comment=comment or f"tell R{agent_reg}, R{data_reg}")

    @staticmethod
    def ask(agent_reg: int, query_reg: int,
            comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.ASK, [agent_reg, query_reg],
                               comment=comment or f"ask R{agent_reg}, R{query_reg}")

    @staticmethod
    def deleg(task_reg: int, target_reg: int,
              comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.DELEG, [task_reg, target_reg],
                               comment=comment or f"deleg R{task_reg}, R{target_reg}")

    @staticmethod
    def bcast(task_reg: int, mask_reg: int,
              comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.BCAST, [task_reg, mask_reg],
                               comment=comment or f"bcast R{task_reg}, R{mask_reg}")

    @staticmethod
    def accept(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.ACCEPT,
                               comment=comment or "accept")

    @staticmethod
    def decline(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.DECLINE,
                               comment=comment or "decline")

    @staticmethod
    def report(status_reg: int, data_reg: int,
               comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.REPORT, [status_reg, data_reg],
                               comment=comment or f"report R{status_reg}, R{data_reg}")

    @staticmethod
    def merge(results_reg: int, strategy_reg: int,
              comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.MERGE, [results_reg, strategy_reg],
                               comment=comment or f"merge R{results_reg}, R{strategy_reg}")

    @staticmethod
    def fork(count_reg: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.FORK, [count_reg],
                               comment=comment or f"fork R{count_reg}")

    @staticmethod
    def join(children_reg: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.JOIN, [children_reg],
                               comment=comment or f"join R{children_reg}")

    @staticmethod
    def signal(name_reg: int, data_reg: int,
               comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.SIGNAL, [name_reg, data_reg],
                               comment=comment or f"signal R{name_reg}, R{data_reg}")

    @staticmethod
    def await_signal(name_reg: int, comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.AWAIT, [name_reg],
                               comment=comment or f"await R{name_reg}")

    @staticmethod
    def trust(agent_reg: int, level_reg: int,
              comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.TRUST, [agent_reg, level_reg],
                               comment=comment or f"trust R{agent_reg}, R{level_reg}")

    @staticmethod
    def discov(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.DISCOV,
                               comment=comment or "discov")

    @staticmethod
    def status(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.STATUS,
                               comment=comment or "status")

    @staticmethod
    def heartbt(comment: str = "") -> "FluxInstruction":
        return FluxInstruction(A2A_Opcodes.HEARTBT,
                               comment=comment or "heartbt")


# =========================================================================
# Register constants
# =========================================================================

R_SCRATCH = 0
R_TASK_START = 1
R_TASK_END = 7
R_AGENT_START = 8
R_AGENT_END = 15


# =========================================================================
# Symbol table for agent name → register mapping
# =========================================================================

@dataclass
class SymbolTable:
    """Manages agent-name-to-register mappings and string pool.

    Attributes:
        agent_map: Maps agent names to register numbers.
        string_pool: Maps string constants to integer IDs.
        next_string_id: Counter for string IDs.
    """
    agent_map: Dict[str, int] = field(default_factory=dict)
    string_pool: Dict[str, int] = field(default_factory=dict)
    next_string_id: int = 0x1000
    _next_agent_reg: int = R_AGENT_START

    def resolve_agent(self, name: str) -> int:
        """Resolve an agent name to a register, allocating if new.

        Args:
            name: Agent name or identifier.

        Returns:
            Register number (R8-R15).
        """
        if name not in self.agent_map:
            if self._next_agent_reg > R_AGENT_END:
                raise ValueError(
                    f"Agent register overflow: cannot allocate for '{name}'. "
                    f"Max {R_AGENT_END - R_AGENT_START + 1} agents."
                )
            self.agent_map[name] = self._next_agent_reg
            self._next_agent_reg += 1
        return self.agent_map[name]

    def intern_string(self, s: str) -> int:
        """Intern a string constant and return its ID.

        Args:
            s: The string to intern.

        Returns:
            Integer ID for the string.
        """
        if s not in self.string_pool:
            self.string_pool[s] = self.next_string_id
            self.next_string_id += 1
        return self.string_pool[s]

    def copy(self) -> "SymbolTable":
        """Create a shallow copy of this symbol table."""
        st = SymbolTable()
        st.agent_map = dict(self.agent_map)
        st.string_pool = dict(self.string_pool)
        st.next_string_id = self.next_string_id
        st._next_agent_reg = self._next_agent_reg
        return st


# =========================================================================
# Strategy encoding for MergeStrategy
# =========================================================================

STRATEGY_CODES: Dict[MergeStrategy, int] = {
    MergeStrategy.SUM: 0x01,
    MergeStrategy.MAX: 0x02,
    MergeStrategy.MIN: 0x03,
    MergeStrategy.VOTE: 0x04,
    MergeStrategy.CONCAT: 0x05,
}


def encode_strategy(strategy: MergeStrategy) -> int:
    """Encode a MergeStrategy as an integer opcode operand."""
    return STRATEGY_CODES[strategy]


def encode_trust_level(level: float) -> int:
    """Encode a trust level (0.0–1.0) as a 16-bit fixed-point integer.

    Uses Q15 fixed-point: value = round(level * 32767).

    Args:
        level: Trust level in [0.0, 1.0].

    Returns:
        16-bit integer encoding.
    """
    clamped = max(0.0, min(1.0, level))
    return round(clamped * 32767)


# =========================================================================
# Code Generator
# =========================================================================

class CodeGenerator:
    """Compiles Coop IR nodes into FLUX bytecode instructions.

    Usage:
        gen = CodeGenerator()
        program = CoopProgram("demo", [Task("t1", "alpha"), ...])
        instructions = gen.generate(program)
    """

    def __init__(self) -> None:
        self.symbols = SymbolTable()
        self.instructions: List[FluxInstruction] = []
        self._task_counter: int = 0
        self._label_counter: int = 0

    def _next_task_reg(self) -> int:
        """Allocate the next available task data register (R1–R7)."""
        reg = R_TASK_START + (self._task_counter % (R_TASK_END - R_TASK_START + 1))
        self._task_counter += 1
        return reg

    def _new_label(self, prefix: str = "L") -> str:
        """Generate a unique label name."""
        label = f"{prefix}_{self._label_counter}"
        self._label_counter += 1
        return label

    def _emit(self, instr: FluxInstruction) -> None:
        """Append an instruction to the output."""
        self.instructions.append(instr)

    def reset(self) -> None:
        """Reset the code generator state for a new compilation."""
        self.symbols = SymbolTable()
        self.instructions = []
        self._task_counter = 0
        self._label_counter = 0

    def generate(self, program: CoopProgram) -> List[FluxInstruction]:
        """Compile a CoopProgram into a list of FLUX instructions.

        Args:
            program: The cooperative program to compile.

        Returns:
            List of compiled FluxInstruction objects.
        """
        self.reset()
        for node in program.nodes:
            self._compile_node(node)
        # Always end with HALT
        self._emit(FluxInstruction.halt("program end"))
        return self.instructions

    def generate_node(self, node: CoopIRNode) -> List[FluxInstruction]:
        """Compile a single IR node into instructions.

        Args:
            node: A single Coop IR node.

        Returns:
            List of compiled FluxInstruction objects.
        """
        saved = list(self.instructions)
        self.instructions = []
        self._compile_node(node)
        result = self.instructions
        self.instructions = saved
        return result

    def _compile_node(self, node: CoopIRNode) -> None:
        """Dispatch compilation for a single IR node."""
        if isinstance(node, Task):
            self._compile_task(node)
        elif isinstance(node, Divide):
            self._compile_divide(node)
        elif isinstance(node, Delegate):
            self._compile_delegate(node)
        elif isinstance(node, Broadcast):
            self._compile_broadcast(node)
        elif isinstance(node, Ask):
            self._compile_ask(node)
        elif isinstance(node, Tell):
            self._compile_tell(node)
        elif isinstance(node, Merge):
            self._compile_merge(node)
        elif isinstance(node, Fork):
            self._compile_fork(node)
        elif isinstance(node, Join):
            self._compile_join(node)
        elif isinstance(node, Trust):
            self._compile_trust(node)
        elif isinstance(node, Report):
            self._compile_report(node)
        elif isinstance(node, Signal):
            self._compile_signal(node)
        elif isinstance(node, Await):
            self._compile_await(node)
        else:
            raise ValueError(f"Unknown IR node type: {type(node).__name__}")

    # ---- Node compilers ----

    def _compile_task(self, task: Task) -> None:
        """Compile a Task node.

        Loads the task payload and agent handle into registers, then
        emits a DELEG instruction to the agent.
        """
        task_reg = self._next_task_reg()
        agent_reg = self.symbols.resolve_agent(task.agent)

        # Load task payload identifier
        payload_id = self.symbols.intern_string(str(task.payload) if task.payload else task.name)
        self._emit(FluxInstruction.movi16(task_reg, payload_id,
                                          f"load task '{task.name}'"))

        # Set priority via MOVI16 to scratch, then DELEG
        if task.priority > 0:
            self._emit(FluxInstruction.movi16(R_SCRATCH, task.priority,
                                              f"priority={task.priority}"))
        self._emit(FluxInstruction.deleg(task_reg, agent_reg,
                                         f"delegate '{task.name}' to {task.agent}"))

    def _compile_divide(self, divide: Divide) -> None:
        """Compile a Divide node.

        Forks N branches (one per subtask), then joins them.
        """
        # Emit parent task setup
        self._compile_task(divide.task)
        # Fork for subtasks
        n = len(divide.subtasks)
        if n > 0:
            self._emit(FluxInstruction.movi16(R_SCRATCH, n,
                                              f"fork {n} subtasks"))
            self._emit(FluxInstruction.fork(R_SCRATCH))
            # Compile each subtask
            for sub in divide.subtasks:
                self._compile_task(sub)
            # Join all
            children_reg = self._next_task_reg()
            self._emit(FluxInstruction.movi16(children_reg, n,
                                              f"join {n} children"))
            self._emit(FluxInstruction.join(children_reg))

    def _compile_delegate(self, delegate: Delegate) -> None:
        """Compile a Delegate node.

        Loads task into a register, resolves the target agent, emits DELEG.
        """
        task_reg = self._next_task_reg()
        agent_reg = self.symbols.resolve_agent(delegate.target_agent)

        payload_id = self.symbols.intern_string(
            str(delegate.task.payload) if delegate.task.payload else delegate.task.name
        )
        self._emit(FluxInstruction.movi16(task_reg, payload_id,
                                          f"load task '{delegate.task.name}'"))
        self._emit(FluxInstruction.deleg(task_reg, agent_reg,
                                         f"delegate to {delegate.target_agent}"))

    def _compile_broadcast(self, bcast: Broadcast) -> None:
        """Compile a Broadcast node.

        Loads task and fleet mask, emits BCAST instruction.
        """
        task_reg = self._next_task_reg()

        payload_id = self.symbols.intern_string(
            str(bcast.task.payload) if bcast.task.payload else bcast.task.name
        )
        self._emit(FluxInstruction.movi16(task_reg, payload_id,
                                          f"load broadcast task '{bcast.task.name}'"))

        # Encode fleet mask
        if isinstance(bcast.fleet_mask, int):
            mask_val = bcast.fleet_mask
        else:
            # List of agent names → encode as bitmask
            mask_val = 0
            for agent_name in bcast.fleet_mask:
                reg = self.symbols.resolve_agent(agent_name)
                mask_val |= (1 << (reg - R_AGENT_START))

        self._emit(FluxInstruction.movi16(R_SCRATCH, mask_val,
                                          f"fleet_mask=0x{mask_val:04X}"))
        self._emit(FluxInstruction.bcast(task_reg, R_SCRATCH,
                                         "broadcast to fleet"))

    def _compile_ask(self, ask: Ask) -> None:
        """Compile an Ask node."""
        agent_reg = self.symbols.resolve_agent(ask.agent)
        query_id = self.symbols.intern_string(str(ask.query))

        self._emit(FluxInstruction.movi16(R_SCRATCH, query_id,
                                          f"load query"))
        self._emit(FluxInstruction.ask(agent_reg, R_SCRATCH,
                                       f"ask {ask.agent}"))

    def _compile_tell(self, tell: Tell) -> None:
        """Compile a Tell node."""
        agent_reg = self.symbols.resolve_agent(tell.agent)
        data_id = self.symbols.intern_string(str(tell.data))

        self._emit(FluxInstruction.movi16(R_SCRATCH, data_id,
                                          f"load data"))
        self._emit(FluxInstruction.tell(agent_reg, R_SCRATCH,
                                        f"tell {tell.agent}"))

    def _compile_merge(self, merge: Merge) -> None:
        """Compile a Merge node.

        Loads results count and strategy code, emits MERGE instruction.
        """
        results_reg = self._next_task_reg()
        strategy_code = encode_strategy(merge.strategy)

        # Load number of results
        n = len(merge.results)
        self._emit(FluxInstruction.movi16(results_reg, n,
                                          f"{n} results"))

        # Load strategy code
        self._emit(FluxInstruction.movi16(R_SCRATCH, strategy_code,
                                          f"strategy={merge.strategy.value}"))

        # Emit MERGE
        self._emit(FluxInstruction.merge(results_reg, R_SCRATCH,
                                         f"merge ({merge.strategy.value})"))

    def _compile_fork(self, fork: Fork) -> None:
        """Compile a Fork node."""
        self._emit(FluxInstruction.movi16(R_SCRATCH, fork.count,
                                          f"fork {fork.count} branches"))
        self._emit(FluxInstruction.fork(R_SCRATCH))

    def _compile_join(self, join: Join) -> None:
        """Compile a Join node."""
        n = len(join.children)
        children_reg = self._next_task_reg()
        self._emit(FluxInstruction.movi16(children_reg, n,
                                          f"join {n} children"))
        self._emit(FluxInstruction.join(children_reg))

    def _compile_trust(self, trust: Trust) -> None:
        """Compile a Trust node.

        Sets trust level using fixed-point encoding.
        """
        agent_reg = self.symbols.resolve_agent(trust.agent)
        level_encoded = encode_trust_level(trust.level)

        self._emit(FluxInstruction.movi16(R_SCRATCH, level_encoded,
                                          f"trust_level={trust.level:.3f}"))
        self._emit(FluxInstruction.trust(agent_reg, R_SCRATCH,
                                         f"trust {trust.agent}={trust.level}"))

    def _compile_report(self, report: Report) -> None:
        """Compile a Report node."""
        status_id = self.symbols.intern_string(report.status)
        data_id = self.symbols.intern_string(str(report.data) if report.data else "")

        status_reg = self._next_task_reg()
        self._emit(FluxInstruction.movi16(status_reg, status_id,
                                          f"status '{report.status}'"))
        self._emit(FluxInstruction.movi16(R_SCRATCH, data_id,
                                          "report data"))
        self._emit(FluxInstruction.report(status_reg, R_SCRATCH,
                                         f"report: {report.status}"))

    def _compile_signal(self, signal: Signal) -> None:
        """Compile a Signal node."""
        name_id = self.symbols.intern_string(signal.name)
        data_id = self.symbols.intern_string(str(signal.data) if signal.data else "")

        self._emit(FluxInstruction.movi16(R_SCRATCH, name_id,
                                          f"signal '{signal.name}'"))
        data_reg = self._next_task_reg()
        self._emit(FluxInstruction.movi16(data_reg, data_id,
                                          "signal data"))
        self._emit(FluxInstruction.signal(R_SCRATCH, data_reg,
                                         f"emit signal '{signal.name}'"))

    def _compile_await(self, await_node: Await) -> None:
        """Compile an Await node."""
        name_id = self.symbols.intern_string(await_node.signal_name)
        self._emit(FluxInstruction.movi16(R_SCRATCH, name_id,
                                          f"await signal '{await_node.signal_name}'"))
        self._emit(FluxInstruction.await_signal(R_SCRATCH))


# =========================================================================
# Optimizer — Peephole optimizations
# =========================================================================

class Optimizer:
    """Peephole optimizer for FLUX bytecode instructions.

    Optimizations:
        1. Eliminate redundant MOV (identity moves: MOV Rn, Rn).
        2. Merge consecutive PUSH/POP pairs on the same register.
        3. Constant folding for MOVI16 chains: MOVI16 Rn, a + MOVI16 Rn, b → MOVI16 Rn, b.
        4. Remove NOP instructions (unless they are the only instruction).
    """

    def __init__(self) -> None:
        self.optimization_count: int = 0

    def reset_stats(self) -> None:
        """Reset optimization statistics."""
        self.optimization_count = 0

    def optimize(self, instructions: List[FluxInstruction]) -> List[FluxInstruction]:
        """Apply all peephole optimizations to the instruction list.

        Optimizations are applied iteratively until no more changes occur
        (fixed-point convergence).

        Args:
            instructions: List of FluxInstruction objects to optimize.

        Returns:
            Optimized list of FluxInstruction objects.
        """
        self.optimization_count = 0
        result = list(instructions)

        changed = True
        max_iterations = 10
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Phase 1: Remove NOPs
            new_result = self._remove_nops(result)
            if len(new_result) != len(result):
                changed = True
                result = new_result

            # Phase 2: Eliminate identity MOV
            new_result = self._eliminate_identity_mov(result)
            if len(new_result) != len(result):
                changed = True
                result = new_result

            # Phase 3: Fold consecutive MOVI16 to same register
            new_result = self._fold_consecutive_movi16(result)
            if len(new_result) != len(result):
                changed = True
                result = new_result

            # Phase 4: Merge PUSH/POP pairs on same register
            new_result = self._merge_push_pop(result)
            if len(new_result) != len(result):
                changed = True
                result = new_result

        return result

    def _remove_nops(self, instructions: List[FluxInstruction]) -> List[FluxInstruction]:
        """Remove NOP instructions.

        Keeps at least one instruction (NOP if that's all there is).
        """
        if len(instructions) <= 1:
            return instructions

        result = []
        for instr in instructions:
            if instr.opcode == SupportOpcodes.NOP:
                self.optimization_count += 1
                continue
            result.append(instr)
        return result if result else instructions

    def _eliminate_identity_mov(self, instructions: List[FluxInstruction]) -> List[FluxInstruction]:
        """Remove MOV Rn, Rn (identity moves)."""
        result = []
        for instr in instructions:
            if (instr.opcode == SupportOpcodes.MOV
                    and len(instr.operands) == 2
                    and instr.operands[0] == instr.operands[1]):
                self.optimization_count += 1
                continue
            result.append(instr)
        return result

    def _fold_consecutive_movi16(self, instructions: List[FluxInstruction]) -> List[FluxInstruction]:
        """If multiple MOVI16 target the same register, keep only the last one."""
        if not instructions:
            return instructions

        result = []
        last_mov16_reg: Dict[int, int] = {}  # reg -> index in result

        for instr in instructions:
            if (instr.opcode == SupportOpcodes.MOVI16
                    and len(instr.operands) >= 1):
                dst = instr.operands[0]
                if dst in last_mov16_reg:
                    # Replace the previous MOVI16 to this register
                    prev_idx = last_mov16_reg[dst]
                    result[prev_idx] = instr
                    self.optimization_count += 1
                    continue
                last_mov16_reg[dst] = len(result)
                result.append(instr)
            else:
                # Non-MOVI16 instruction: clear tracking for registers
                # that don't have dependencies after this point
                # (conservative: only clear for registers used as source)
                if instr.opcode == SupportOpcodes.MOV and len(instr.operands) == 2:
                    # MOV reads from src; clear dst tracking
                    src = instr.operands[1]
                    last_mov16_reg.pop(src, None)
                result.append(instr)

        return result

    def _merge_push_pop(self, instructions: List[FluxInstruction]) -> List[FluxInstruction]:
        """Remove PUSH Rn / POP Rn pairs on the same register (no-op pair)."""
        result = []
        i = 0
        while i < len(instructions):
            if (i + 1 < len(instructions)
                    and instructions[i].opcode == SupportOpcodes.PUSH
                    and instructions[i + 1].opcode == SupportOpcodes.POP
                    and len(instructions[i].operands) >= 1
                    and len(instructions[i + 1].operands) >= 1
                    and instructions[i].operands[0] == instructions[i + 1].operands[0]):
                # PUSH Rn / POP Rn → eliminate both
                self.optimization_count += 2
                i += 2
                continue
            result.append(instructions[i])
            i += 1
        return result


# =========================================================================
# Pattern Library — Pre-built cooperative patterns
# =========================================================================

class PatternLibrary:
    """Pre-built cooperative intelligence patterns compiled to Coop IR.

    Patterns:
        - map_reduce: Classic map-reduce across workers.
        - scatter_gather: Scatter task to agents, gather results.
        - consensus_round: Weighted voting among agents.
        - pipeline: Sequential processing across stages.
        - retry_with_backoff: Resilient execution with exponential backoff.
    """

    @staticmethod
    def map_reduce(
        map_fn_name: str,
        reduce_fn_name: str,
        inputs: List[Any],
        workers: List[str],
    ) -> CoopProgram:
        """Build a map-reduce cooperative program.

        Distributes inputs across workers, collects results, and merges
        them using the specified reduce strategy.

        Args:
            map_fn_name: Name/identifier for the map function.
            reduce_fn_name: Name/identifier for the reduce function.
            inputs: List of input data items to process.
            workers: List of agent names to distribute work across.

        Returns:
            CoopProgram representing the map-reduce pattern.
        """
        prog = CoopProgram(f"map_reduce_{map_fn_name}")

        # Phase 1: Fork workers
        prog.add(Fork(state="map_reduce", count=len(workers)))

        # Phase 2: Scatter inputs (delegate each input to a worker)
        for i, inp in enumerate(inputs):
            worker = workers[i % len(workers)]
            prog.add(Delegate(
                task=Task(name=f"map_{i}", agent=worker, payload=inp),
                target_agent=worker,
            ))

        # Phase 3: Join all workers
        prog.add(Join(children=workers))

        # Phase 4: Merge results
        prog.add(Merge(
            results=[f"result_{i}" for i in range(len(inputs))],
            strategy=MergeStrategy.SUM,
        ))

        return prog

    @staticmethod
    def scatter_gather(
        task_name: str,
        task_payload: Any,
        agents: List[str],
    ) -> CoopProgram:
        """Build a scatter-gather cooperative program.

        Scatters a task to multiple agents, then gathers all results.

        Args:
            task_name: Name of the task to scatter.
            task_payload: Payload data for the task.
            agents: List of agent names to scatter to.

        Returns:
            CoopProgram representing the scatter-gather pattern.
        """
        prog = CoopProgram(f"scatter_gather_{task_name}")

        # Broadcast to all agents
        prog.add(Broadcast(
            task=Task(name=task_name, agent="fleet", payload=task_payload),
            fleet_mask=list(agents),
        ))

        # Await results from each agent
        for agent in agents:
            prog.add(Await(signal_name=f"result_{agent}"))

        # Merge all results
        prog.add(Merge(
            results=[f"result_{a}" for a in agents],
            strategy=MergeStrategy.CONCAT,
        ))

        return prog

    @staticmethod
    def consensus_round(
        proposal: str,
        voters: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> CoopProgram:
        """Build a consensus (weighted voting) cooperative program.

        Each voter evaluates the proposal, then results are merged via
        VOTE strategy.

        Args:
            proposal: The proposal to vote on.
            voters: List of agent names that will vote.
            weights: Optional per-agent vote weights.

        Returns:
            CoopProgram representing the consensus pattern.
        """
        prog = CoopProgram(f"consensus_{proposal[:20]}")

        # Set trust levels based on weights (if provided)
        if weights:
            for agent, weight in weights.items():
                prog.add(Trust(agent=agent, level=weight))

        # Broadcast proposal to all voters
        prog.add(Broadcast(
            task=Task(name="vote_request", agent="fleet", payload=proposal),
            fleet_mask=list(voters),
        ))

        # Await votes
        for voter in voters:
            prog.add(Await(signal_name=f"vote_{voter}"))

        # Merge via VOTE strategy
        prog.add(Merge(
            results=[f"vote_{v}" for v in voters],
            strategy=MergeStrategy.VOTE,
        ))

        return prog

    @staticmethod
    def pipeline(
        stages: List[str],
        agent_per_stage: List[str],
    ) -> CoopProgram:
        """Build a pipeline cooperative program.

        Sequential processing where each stage is handled by a
        different agent.

        Args:
            stages: List of stage names.
            agent_per_stage: Agent name for each stage (same length as stages).

        Returns:
            CoopProgram representing the pipeline pattern.
        """
        prog = CoopProgram("pipeline")

        if len(stages) != len(agent_per_stage):
            raise ValueError(
                f"stages ({len(stages)}) and agent_per_stage "
                f"({len(agent_per_stage)}) must have the same length"
            )

        # Fork for parallel potential at each stage boundary
        prog.add(Fork(state="pipeline", count=len(stages)))

        # Sequential stage execution
        for i, (stage_name, agent) in enumerate(zip(stages, agent_per_stage)):
            prog.add(Delegate(
                task=Task(name=stage_name, agent=agent,
                          payload=f"stage_{i}"),
                target_agent=agent,
            ))
            # Signal completion of this stage
            prog.add(Signal(name=f"stage_done_{i}", data=stage_name))

        # Join all stages
        prog.add(Join(children=[f"stage_{i}" for i in range(len(stages))]))

        return prog

    @staticmethod
    def retry_with_backoff(
        task_name: str,
        max_retries: int,
        base_delay: float,
        agent: str = "worker",
    ) -> CoopProgram:
        """Build a retry-with-exponential-backoff cooperative program.

        Retries a task up to max_retries times with exponential backoff
        between attempts.

        Args:
            task_name: Name of the task to retry.
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay (used to compute backoff).
            agent: Agent to delegate the task to.

        Returns:
            CoopProgram representing the retry pattern.
        """
        prog = CoopProgram(f"retry_{task_name}")

        for attempt in range(max_retries):
            # Report attempt
            prog.add(Report(
                status=f"attempt_{attempt}",
                data={"attempt": attempt, "task": task_name},
            ))

            # Delegate the task
            prog.add(Delegate(
                task=Task(name=f"{task_name}_attempt_{attempt}", agent=agent),
                target_agent=agent,
            ))

            # Await result signal
            prog.add(Await(signal_name=f"result_{task_name}_attempt_{attempt}"))

            # If not last attempt, signal retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                prog.add(Signal(
                    name="retry_schedule",
                    data={"delay": delay, "next_attempt": attempt + 1},
                ))

        # Final merge
        prog.add(Merge(
            results=[f"result_{task_name}_attempt_{i}" for i in range(max_retries)],
            strategy=MergeStrategy.MAX,
        ))

        return prog


# =========================================================================
# High-level compiler interface
# =========================================================================

def compile_coop_program(
    program: CoopProgram,
    optimize: bool = True,
) -> List[FluxInstruction]:
    """Compile a CoopProgram to optimized FLUX bytecode.

    This is the main entry point for the cooperative intelligence compiler.

    Args:
        program: The cooperative program to compile.
        optimize: Whether to run the peephole optimizer on the output.

    Returns:
        List of compiled (and optionally optimized) FluxInstruction objects.
    """
    gen = CodeGenerator()
    instructions = gen.generate(program)

    if optimize:
        opt = Optimizer()
        instructions = opt.optimize(instructions)

    return instructions
