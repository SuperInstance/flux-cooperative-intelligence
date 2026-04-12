"""
Comprehensive tests for the Cooperative Intelligence Compiler.

Covers:
    - IR node construction and attributes
    - Opcode definitions and values
    - FluxInstruction factory methods and serialization
    - SymbolTable (agent resolution, string interning, copy)
    - Strategy and trust-level encoding
    - CodeGenerator (all 13 IR node types)
    - CodeGenerator edge cases
    - Optimizer (all 4 peephole passes + fixed-point)
    - PatternLibrary (all 5 patterns)
    - compile_coop_program high-level interface
    - Integration scenarios
"""

import math
import pytest

from coop_compiler.compiler import (
    A2A_Opcodes,
    SupportOpcodes,
    MergeStrategy,
    Task,
    Divide,
    Delegate,
    Broadcast,
    Ask,
    Tell,
    Merge,
    Fork,
    Join,
    Trust,
    Report,
    Signal,
    Await,
    CoopProgram,
    FluxInstruction,
    SymbolTable,
    CodeGenerator,
    Optimizer,
    PatternLibrary,
    STRATEGY_CODES,
    R_SCRATCH,
    R_TASK_START,
    R_TASK_END,
    R_AGENT_START,
    R_AGENT_END,
    compile_coop_program,
    encode_strategy,
    encode_trust_level,
)


# =========================================================================
# 1. Opcode definitions
# =========================================================================

class TestOpcodes:
    """Tests for A2A and supporting opcode definitions."""

    def test_tell_opcode_value(self):
        assert A2A_Opcodes.TELL == 0x50

    def test_ask_opcode_value(self):
        assert A2A_Opcodes.ASK == 0x51

    def test_deleg_opcode_value(self):
        assert A2A_Opcodes.DELEG == 0x52

    def test_bcast_opcode_value(self):
        assert A2A_Opcodes.BCAST == 0x53

    def test_accept_opcode_value(self):
        assert A2A_Opcodes.ACCEPT == 0x54

    def test_decline_opcode_value(self):
        assert A2A_Opcodes.DECLINE == 0x55

    def test_report_opcode_value(self):
        assert A2A_Opcodes.REPORT == 0x56

    def test_merge_opcode_value(self):
        assert A2A_Opcodes.MERGE == 0x57

    def test_fork_opcode_value(self):
        assert A2A_Opcodes.FORK == 0x58

    def test_join_opcode_value(self):
        assert A2A_Opcodes.JOIN == 0x59

    def test_signal_opcode_value(self):
        assert A2A_Opcodes.SIGNAL == 0x5A

    def test_await_opcode_value(self):
        assert A2A_Opcodes.AWAIT == 0x5B

    def test_trust_opcode_value(self):
        assert A2A_Opcodes.TRUST == 0x5C

    def test_discov_opcode_value(self):
        assert A2A_Opcodes.DISCOV == 0x5D

    def test_status_opcode_value(self):
        assert A2A_Opcodes.STATUS == 0x5E

    def test_heartbt_opcode_value(self):
        assert A2A_Opcodes.HEARTBT == 0x5F

    def test_all_a2a_opcodes_distinct(self):
        codes = [op.value for op in A2A_Opcodes]
        assert len(codes) == len(set(codes))

    def test_support_halt_value(self):
        assert SupportOpcodes.HALT == 0x00

    def test_support_nop_value(self):
        assert SupportOpcodes.NOP == 0x01

    def test_support_mov_value(self):
        assert SupportOpcodes.MOV == 0x3A

    def test_support_movi16_value(self):
        assert SupportOpcodes.MOVI16 == 0x40

    def test_support_add_value(self):
        assert SupportOpcodes.ADD == 0x20

    def test_support_cmp_eq_value(self):
        assert SupportOpcodes.CMP_EQ == 0x2C

    def test_support_jnz_value(self):
        assert SupportOpcodes.JNZ == 0x3D

    def test_support_push_value(self):
        assert SupportOpcodes.PUSH == 0x0C

    def test_support_pop_value(self):
        assert SupportOpcodes.POP == 0x0D


# =========================================================================
# 2. IR node construction
# =========================================================================

class TestIRNodes:
    """Tests for Coop IR data classes."""

    def test_task_defaults(self):
        t = Task(name="t1", agent="alpha")
        assert t.name == "t1"
        assert t.agent == "alpha"
        assert t.payload is None
        assert t.priority == 0

    def test_task_full(self):
        t = Task(name="t2", agent="beta", payload={"key": "val"}, priority=5)
        assert t.payload == {"key": "val"}
        assert t.priority == 5

    def test_divide(self):
        sub1 = Task(name="s1", agent="a")
        sub2 = Task(name="s2", agent="b")
        parent = Task(name="parent", agent="owner")
        d = Divide(task=parent, subtasks=[sub1, sub2])
        assert len(d.subtasks) == 2
        assert d.subtasks[0].name == "s1"

    def test_divide_empty_subtasks(self):
        d = Divide(task=Task(name="p", agent="o"))
        assert d.subtasks == []

    def test_delegate(self):
        t = Task(name="t", agent="a")
        d = Delegate(task=t, target_agent="beta")
        assert d.target_agent == "beta"

    def test_broadcast_int_mask(self):
        b = Broadcast(task=Task(name="t", agent="fleet"), fleet_mask=0xFF)
        assert b.fleet_mask == 0xFF

    def test_broadcast_list_mask(self):
        b = Broadcast(
            task=Task(name="t", agent="fleet"),
            fleet_mask=["alpha", "beta"],
        )
        assert b.fleet_mask == ["alpha", "beta"]

    def test_ask(self):
        a = Ask(agent="alpha", query="what is the answer?")
        assert a.agent == "alpha"
        assert a.query == "what is the answer?"

    def test_tell(self):
        t = Tell(agent="alpha", data={"result": 42})
        assert t.data == {"result": 42}

    def test_merge_default_strategy(self):
        m = Merge(results=[1, 2, 3])
        assert m.strategy == MergeStrategy.SUM

    def test_merge_custom_strategy(self):
        m = Merge(results=[1, 2, 3], strategy=MergeStrategy.VOTE)
        assert m.strategy == MergeStrategy.VOTE

    def test_fork(self):
        f = Fork(state="shared", count=4)
        assert f.count == 4

    def test_fork_defaults(self):
        f = Fork()
        assert f.count == 2
        assert f.state is None

    def test_join(self):
        j = Join(children=["c1", "c2", "c3"])
        assert len(j.children) == 3

    def test_join_empty(self):
        j = Join()
        assert j.children == []

    def test_trust(self):
        t = Trust(agent="alpha", level=0.9)
        assert t.level == 0.9

    def test_trust_default(self):
        t = Trust(agent="alpha")
        assert t.level == 0.5

    def test_report(self):
        r = Report(status="ok", data={"progress": 50})
        assert r.status == "ok"

    def test_signal(self):
        s = Signal(name="done", data="payload")
        assert s.name == "done"

    def test_await(self):
        a = Await(signal_name="done")
        assert a.signal_name == "done"

    def test_coop_program_add_fluent(self):
        prog = CoopProgram("test")
        prog.add(Task("t1", "a")).add(Task("t2", "b"))
        assert len(prog.nodes) == 2

    def test_coop_program_empty(self):
        prog = CoopProgram("empty")
        assert prog.nodes == []

    def test_merge_strategy_values(self):
        assert MergeStrategy.SUM.value == "SUM"
        assert MergeStrategy.MAX.value == "MAX"
        assert MergeStrategy.MIN.value == "MIN"
        assert MergeStrategy.VOTE.value == "VOTE"
        assert MergeStrategy.CONCAT.value == "CONCAT"


# =========================================================================
# 3. FluxInstruction
# =========================================================================

class TestFluxInstruction:
    """Tests for FluxInstruction construction, factory methods, and serialization."""

    def test_basic_instruction(self):
        instr = FluxInstruction(opcode=0x50, operands=[1, 2])
        assert instr.opcode == 0x50
        assert instr.operands == [1, 2]

    def test_instruction_defaults(self):
        instr = FluxInstruction(opcode=0x00)
        assert instr.operands == []
        assert instr.comment == ""
        assert instr.label == ""

    def test_instruction_with_label(self):
        instr = FluxInstruction(opcode=0x00, label="start")
        assert instr.label == "start"

    def test_repr_with_label(self):
        instr = FluxInstruction(opcode=0x50, operands=[1, 2], label="L0")
        r = repr(instr)
        assert "L0:" in r
        assert "0x50" in r

    def test_repr_with_comment(self):
        instr = FluxInstruction(opcode=0x50, operands=[1, 2], comment="tell")
        r = repr(instr)
        assert "; tell" in r

    def test_to_bytes_single_byte_opcode(self):
        instr = FluxInstruction(opcode=0x50, operands=[])
        b = instr.to_bytes()
        assert b == b"\x50"

    def test_to_bytes_with_byte_operands(self):
        instr = FluxInstruction(opcode=0x50, operands=[1, 2])
        b = instr.to_bytes()
        assert b == b"\x50\x01\x02"

    def test_to_bytes_with_16bit_operand(self):
        instr = FluxInstruction(opcode=0x40, operands=[1, 0x1000])
        b = instr.to_bytes()
        assert len(b) == 4  # 1 opcode + 1 byte reg + 2 bytes imm16
        assert b[0] == 0x40
        assert b[1] == 0x01
        assert b[2:] == b"\x10\x00"

    def test_factory_halt(self):
        instr = FluxInstruction.halt()
        assert instr.opcode == SupportOpcodes.HALT

    def test_factory_nop(self):
        instr = FluxInstruction.nop()
        assert instr.opcode == SupportOpcodes.NOP

    def test_factory_mov(self):
        instr = FluxInstruction.mov(1, 2)
        assert instr.opcode == SupportOpcodes.MOV
        assert instr.operands == [1, 2]

    def test_factory_movi16(self):
        instr = FluxInstruction.movi16(1, 42)
        assert instr.opcode == SupportOpcodes.MOVI16
        assert instr.operands == [1, 42]

    def test_factory_add(self):
        instr = FluxInstruction.add(0, 1, 2)
        assert instr.opcode == SupportOpcodes.ADD
        assert instr.operands == [0, 1, 2]

    def test_factory_tell(self):
        instr = FluxInstruction.tell(8, 1)
        assert instr.opcode == A2A_Opcodes.TELL
        assert instr.operands == [8, 1]

    def test_factory_ask(self):
        instr = FluxInstruction.ask(8, 0)
        assert instr.opcode == A2A_Opcodes.ASK

    def test_factory_deleg(self):
        instr = FluxInstruction.deleg(1, 8)
        assert instr.opcode == A2A_Opcodes.DELEG
        assert instr.operands == [1, 8]

    def test_factory_bcast(self):
        instr = FluxInstruction.bcast(1, 0)
        assert instr.opcode == A2A_Opcodes.BCAST

    def test_factory_accept(self):
        instr = FluxInstruction.accept()
        assert instr.opcode == A2A_Opcodes.ACCEPT
        assert instr.operands == []

    def test_factory_decline(self):
        instr = FluxInstruction.decline()
        assert instr.opcode == A2A_Opcodes.DECLINE

    def test_factory_report(self):
        instr = FluxInstruction.report(1, 0)
        assert instr.opcode == A2A_Opcodes.REPORT
        assert instr.operands == [1, 0]

    def test_factory_merge(self):
        instr = FluxInstruction.merge(1, 0)
        assert instr.opcode == A2A_Opcodes.MERGE
        assert instr.operands == [1, 0]

    def test_factory_fork(self):
        instr = FluxInstruction.fork(0)
        assert instr.opcode == A2A_Opcodes.FORK

    def test_factory_join(self):
        instr = FluxInstruction.join(1)
        assert instr.opcode == A2A_Opcodes.JOIN

    def test_factory_signal(self):
        instr = FluxInstruction.signal(0, 1)
        assert instr.opcode == A2A_Opcodes.SIGNAL

    def test_factory_await(self):
        instr = FluxInstruction.await_signal(0)
        assert instr.opcode == A2A_Opcodes.AWAIT
        assert instr.operands == [0]

    def test_factory_trust(self):
        instr = FluxInstruction.trust(8, 0)
        assert instr.opcode == A2A_Opcodes.TRUST

    def test_factory_discov(self):
        instr = FluxInstruction.discov()
        assert instr.opcode == A2A_Opcodes.DISCOV

    def test_factory_status(self):
        instr = FluxInstruction.status()
        assert instr.opcode == A2A_Opcodes.STATUS

    def test_factory_heartbt(self):
        instr = FluxInstruction.heartbt()
        assert instr.opcode == A2A_Opcodes.HEARTBT

    def test_factory_push(self):
        instr = FluxInstruction.push(1)
        assert instr.opcode == SupportOpcodes.PUSH

    def test_factory_pop(self):
        instr = FluxInstruction.pop(1)
        assert instr.opcode == SupportOpcodes.POP


# =========================================================================
# 4. SymbolTable
# =========================================================================

class TestSymbolTable:
    """Tests for the SymbolTable agent resolution and string interning."""

    def test_resolve_agent_allocates(self):
        st = SymbolTable()
        reg = st.resolve_agent("alpha")
        assert reg == R_AGENT_START  # 8

    def test_resolve_agent_second(self):
        st = SymbolTable()
        st.resolve_agent("alpha")
        reg = st.resolve_agent("beta")
        assert reg == R_AGENT_START + 1  # 9

    def test_resolve_agent_caches(self):
        st = SymbolTable()
        r1 = st.resolve_agent("alpha")
        r2 = st.resolve_agent("alpha")
        assert r1 == r2

    def test_resolve_agent_overflow(self):
        st = SymbolTable()
        # 8 agents (R8-R15)
        for i in range(8):
            st.resolve_agent(f"agent_{i}")
        with pytest.raises(ValueError, match="Agent register overflow"):
            st.resolve_agent("agent_overflow")

    def test_intern_string(self):
        st = SymbolTable()
        sid = st.intern_string("hello")
        assert sid == 0x1000

    def test_intern_string_caches(self):
        st = SymbolTable()
        s1 = st.intern_string("hello")
        s2 = st.intern_string("hello")
        assert s1 == s2

    def test_intern_string_increments(self):
        st = SymbolTable()
        s1 = st.intern_string("a")
        s2 = st.intern_string("b")
        assert s2 == s1 + 1

    def test_copy_independence(self):
        st = SymbolTable()
        st.resolve_agent("alpha")
        st2 = st.copy()
        st2.resolve_agent("beta")
        assert "beta" not in st.agent_map
        assert "beta" in st2.agent_map

    def test_copy_preserves_mappings(self):
        st = SymbolTable()
        st.resolve_agent("alpha")
        st.intern_string("hello")
        st2 = st.copy()
        assert st2.resolve_agent("alpha") == R_AGENT_START
        assert st2.intern_string("hello") == 0x1000


# =========================================================================
# 5. Encoding helpers
# =========================================================================

class TestEncoding:
    """Tests for strategy and trust level encoding."""

    def test_encode_strategy_sum(self):
        assert encode_strategy(MergeStrategy.SUM) == STRATEGY_CODES[MergeStrategy.SUM]

    def test_encode_strategy_max(self):
        assert encode_strategy(MergeStrategy.MAX) == STRATEGY_CODES[MergeStrategy.MAX]

    def test_encode_strategy_min(self):
        assert encode_strategy(MergeStrategy.MIN) == STRATEGY_CODES[MergeStrategy.MIN]

    def test_encode_strategy_vote(self):
        assert encode_strategy(MergeStrategy.VOTE) == STRATEGY_CODES[MergeStrategy.VOTE]

    def test_encode_strategy_concat(self):
        assert encode_strategy(MergeStrategy.CONCAT) == STRATEGY_CODES[MergeStrategy.CONCAT]

    def test_encode_strategy_all_distinct(self):
        codes = [encode_strategy(s) for s in MergeStrategy]
        assert len(codes) == len(set(codes))

    def test_encode_trust_zero(self):
        assert encode_trust_level(0.0) == 0

    def test_encode_trust_one(self):
        assert encode_trust_level(1.0) == 32767

    def test_encode_trust_half(self):
        val = encode_trust_level(0.5)
        assert 16000 <= val <= 17000  # approximately 16383

    def test_encode_trust_clamp_high(self):
        assert encode_trust_level(2.0) == 32767

    def test_encode_trust_clamp_low(self):
        assert encode_trust_level(-1.0) == 0

    def test_encode_trust_rounds(self):
        val = encode_trust_level(1.0 / 3.0)
        assert val == round(32767 / 3.0)


# =========================================================================
# 6. CodeGenerator — individual node compilation
# =========================================================================

class TestCodeGeneratorTask:
    """Tests for Task node compilation."""

    def test_task_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Task("t1", "alpha"))
        assert len(instrs) >= 2  # MOVI16 + DELEG
        assert instrs[0].opcode == SupportOpcodes.MOVI16
        deleg = [i for i in instrs if i.opcode == A2A_Opcodes.DELEG]
        assert len(deleg) == 1

    def test_task_with_priority(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Task("t1", "alpha", priority=3))
        movi = [i for i in instrs if i.opcode == SupportOpcodes.MOVI16]
        assert len(movi) >= 2  # payload + priority

    def test_task_registers(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Task("t1", "alpha"))
        deleg = next(i for i in instrs if i.opcode == A2A_Opcodes.DELEG)
        assert deleg.operands[0] >= R_TASK_START
        assert deleg.operands[0] <= R_TASK_END
        assert deleg.operands[1] == R_AGENT_START


class TestCodeGeneratorDivide:
    """Tests for Divide node compilation."""

    def test_divide_basic(self):
        sub1 = Task("s1", "a")
        sub2 = Task("s2", "b")
        parent = Task("parent", "owner")
        gen = CodeGenerator()
        instrs = gen.generate_node(Divide(task=parent, subtasks=[sub1, sub2]))
        assert any(i.opcode == A2A_Opcodes.FORK for i in instrs)
        assert any(i.opcode == A2A_Opcodes.JOIN for i in instrs)

    def test_divide_empty_subtasks(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Divide(task=Task("p", "o")))
        # With empty subtasks, no fork/join should be emitted
        fork_instrs = [i for i in instrs if i.opcode == A2A_Opcodes.FORK]
        assert len(fork_instrs) == 0


class TestCodeGeneratorDelegate:
    """Tests for Delegate node compilation."""

    def test_delegate_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(
            Delegate(task=Task("t", "a"), target_agent="beta")
        )
        deleg = next(i for i in instrs if i.opcode == A2A_Opcodes.DELEG)
        # target_agent "beta" is the first agent resolved → R_AGENT_START
        assert deleg.operands[1] == R_AGENT_START

    def test_delegate_same_agent(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(
            Delegate(task=Task("t", "alpha"), target_agent="alpha")
        )
        deleg = next(i for i in instrs if i.opcode == A2A_Opcodes.DELEG)
        assert deleg.operands[1] == R_AGENT_START


class TestCodeGeneratorBroadcast:
    """Tests for Broadcast node compilation."""

    def test_broadcast_int_mask(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(
            Broadcast(task=Task("t", "fleet"), fleet_mask=0xFF)
        )
        bcast = next(i for i in instrs if i.opcode == A2A_Opcodes.BCAST)
        assert bcast is not None

    def test_broadcast_list_mask(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(
            Broadcast(task=Task("t", "fleet"), fleet_mask=["a", "b"])
        )
        bcast = next(i for i in instrs if i.opcode == A2A_Opcodes.BCAST)
        assert bcast is not None


class TestCodeGeneratorAsk:
    """Tests for Ask node compilation."""

    def test_ask_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Ask(agent="alpha", query="test query"))
        ask = next(i for i in instrs if i.opcode == A2A_Opcodes.ASK)
        assert ask.operands[0] == R_AGENT_START


class TestCodeGeneratorTell:
    """Tests for Tell node compilation."""

    def test_tell_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Tell(agent="alpha", data="some data"))
        tell = next(i for i in instrs if i.opcode == A2A_Opcodes.TELL)
        assert tell.operands[0] == R_AGENT_START


class TestCodeGeneratorMerge:
    """Tests for Merge node compilation."""

    def test_merge_sum(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Merge(results=[1, 2, 3], strategy=MergeStrategy.SUM))
        merge = next(i for i in instrs if i.opcode == A2A_Opcodes.MERGE)
        assert merge is not None

    def test_merge_vote(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Merge(results=[1], strategy=MergeStrategy.VOTE))
        merge = next(i for i in instrs if i.opcode == A2A_Opcodes.MERGE)
        assert merge is not None

    def test_merge_empty_results(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Merge(results=[], strategy=MergeStrategy.SUM))
        merge = next(i for i in instrs if i.opcode == A2A_Opcodes.MERGE)
        assert merge is not None


class TestCodeGeneratorForkJoin:
    """Tests for Fork and Join node compilation."""

    def test_fork_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Fork(count=4))
        fork = next(i for i in instrs if i.opcode == A2A_Opcodes.FORK)
        assert fork is not None

    def test_join_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Join(children=["a", "b"]))
        join = next(i for i in instrs if i.opcode == A2A_Opcodes.JOIN)
        assert join is not None

    def test_join_empty_children(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Join())
        join = next(i for i in instrs if i.opcode == A2A_Opcodes.JOIN)
        assert join is not None


class TestCodeGeneratorTrust:
    """Tests for Trust node compilation."""

    def test_trust_high(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Trust(agent="alpha", level=0.9))
        trust = next(i for i in instrs if i.opcode == A2A_Opcodes.TRUST)
        assert trust.operands[0] == R_AGENT_START

    def test_trust_zero(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Trust(agent="alpha", level=0.0))
        trust = next(i for i in instrs if i.opcode == A2A_Opcodes.TRUST)
        assert trust is not None


class TestCodeGeneratorReport:
    """Tests for Report node compilation."""

    def test_report_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Report(status="ok"))
        report = next(i for i in instrs if i.opcode == A2A_Opcodes.REPORT)
        assert report is not None

    def test_report_with_data(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Report(status="ok", data="payload"))
        report = next(i for i in instrs if i.opcode == A2A_Opcodes.REPORT)
        assert report is not None


class TestCodeGeneratorSignalAwait:
    """Tests for Signal and Await node compilation."""

    def test_signal_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Signal(name="done", data="ok"))
        signal = next(i for i in instrs if i.opcode == A2A_Opcodes.SIGNAL)
        assert signal is not None

    def test_await_basic(self):
        gen = CodeGenerator()
        instrs = gen.generate_node(Await(signal_name="done"))
        await_instr = next(i for i in instrs if i.opcode == A2A_Opcodes.AWAIT)
        assert await_instr is not None


# =========================================================================
# 7. CodeGenerator — full program compilation
# =========================================================================

class TestCodeGeneratorProgram:
    """Tests for full CoopProgram compilation."""

    def test_empty_program_generates_halt(self):
        gen = CodeGenerator()
        prog = CoopProgram("empty")
        instrs = gen.generate(prog)
        assert len(instrs) >= 1
        assert instrs[-1].opcode == SupportOpcodes.HALT

    def test_single_task_program(self):
        gen = CodeGenerator()
        prog = CoopProgram("single", [Task("t1", "alpha")])
        instrs = gen.generate(prog)
        assert instrs[-1].opcode == SupportOpcodes.HALT
        deleg = [i for i in instrs if i.opcode == A2A_Opcodes.DELEG]
        assert len(deleg) == 1

    def test_multi_node_program(self):
        gen = CodeGenerator()
        prog = CoopProgram("multi", [
            Task("t1", "alpha"),
            Task("t2", "beta"),
            Report(status="done"),
        ])
        instrs = gen.generate(prog)
        deleg = [i for i in instrs if i.opcode == A2A_Opcodes.DELEG]
        assert len(deleg) == 2
        reports = [i for i in instrs if i.opcode == A2A_Opcodes.REPORT]
        assert len(reports) == 1

    def test_reset_clears_state(self):
        gen = CodeGenerator()
        gen.generate(CoopProgram("first", [Task("t1", "alpha")]))
        assert len(gen.symbols.agent_map) > 0
        gen.reset()
        assert len(gen.symbols.agent_map) == 0

    def test_unknown_node_raises(self):
        gen = CodeGenerator()
        prog = CoopProgram("bad")
        # Manually append something that's not a valid node
        prog.nodes.append("not a node")  # type: ignore
        with pytest.raises(ValueError, match="Unknown IR node type"):
            gen.generate(prog)


# =========================================================================
# 8. Optimizer
# =========================================================================

class TestOptimizer:
    """Tests for all peephole optimization passes."""

    def test_remove_nops(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.nop(),
            FluxInstruction.halt(),
            FluxInstruction.nop(),
        ]
        result = opt.optimize(instrs)
        nops = [i for i in result if i.opcode == SupportOpcodes.NOP]
        assert len(nops) == 0
        assert any(i.opcode == SupportOpcodes.HALT for i in result)

    def test_remove_nops_keeps_single_nop(self):
        opt = Optimizer()
        instrs = [FluxInstruction.nop()]
        result = opt.optimize(instrs)
        assert len(result) == 1

    def test_eliminate_identity_mov(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.mov(1, 1),  # identity
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        movs = [i for i in result if i.opcode == SupportOpcodes.MOV]
        assert len(movs) == 0

    def test_keeps_non_identity_mov(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.mov(1, 2),
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        movs = [i for i in result if i.opcode == SupportOpcodes.MOV]
        assert len(movs) == 1

    def test_fold_consecutive_movi16(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.movi16(1, 100),
            FluxInstruction.movi16(1, 200),  # overwrites previous
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        movi16s = [i for i in result if i.opcode == SupportOpcodes.MOVI16]
        assert len(movi16s) == 1
        assert movi16s[0].operands[1] == 200

    def test_keeps_movi16_different_regs(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.movi16(1, 100),
            FluxInstruction.movi16(2, 200),
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        movi16s = [i for i in result if i.opcode == SupportOpcodes.MOVI16]
        assert len(movi16s) == 2

    def test_merge_push_pop_same_reg(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.push(1),
            FluxInstruction.pop(1),  # no-op pair
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        pushs = [i for i in result if i.opcode == SupportOpcodes.PUSH]
        pops = [i for i in result if i.opcode == SupportOpcodes.POP]
        assert len(pushs) == 0
        assert len(pops) == 0

    def test_keeps_push_pop_different_reg(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.push(1),
            FluxInstruction.pop(2),
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        pushs = [i for i in result if i.opcode == SupportOpcodes.PUSH]
        pops = [i for i in result if i.opcode == SupportOpcodes.POP]
        assert len(pushs) == 1
        assert len(pops) == 1

    def test_combined_optimizations(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.nop(),
            FluxInstruction.mov(1, 1),
            FluxInstruction.movi16(1, 10),
            FluxInstruction.movi16(1, 20),
            FluxInstruction.push(3),
            FluxInstruction.pop(3),
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        # NOP, identity MOV, folded MOVI16, PUSH/POP pair all eliminated
        assert len(result) < len(instrs)
        assert result[-1].opcode == SupportOpcodes.HALT

    def test_optimization_count(self):
        opt = Optimizer()
        instrs = [
            FluxInstruction.nop(),
            FluxInstruction.mov(1, 1),
            FluxInstruction.halt(),
        ]
        opt.optimize(instrs)
        assert opt.optimization_count >= 2

    def test_optimize_empty_list(self):
        opt = Optimizer()
        result = opt.optimize([])
        assert result == []

    def test_reset_stats(self):
        opt = Optimizer()
        # Need >1 instruction so NOP removal kicks in
        opt.optimize([FluxInstruction.nop(), FluxInstruction.halt()])
        assert opt.optimization_count > 0
        opt.reset_stats()
        assert opt.optimization_count == 0

    def test_fixed_point_convergence(self):
        """Test that multiple passes are applied until fixed point."""
        opt = Optimizer()
        instrs = [
            # Chain that requires two passes:
            # Pass 1: remove NOP
            # Pass 1: fold MOVI16 (but MOV still present)
            # After: identity MOV is now adjacent and can be removed
            FluxInstruction.nop(),
            FluxInstruction.movi16(1, 10),
            FluxInstruction.movi16(1, 20),
            FluxInstruction.mov(2, 2),
            FluxInstruction.halt(),
        ]
        result = opt.optimize(instrs)
        assert len(result) < len(instrs)
        assert result[-1].opcode == SupportOpcodes.HALT


# =========================================================================
# 9. PatternLibrary
# =========================================================================

class TestPatternMapReduce:
    """Tests for the map_reduce pattern."""

    def test_map_reduce_basic(self):
        prog = PatternLibrary.map_reduce(
            map_fn_name="analyze",
            reduce_fn_name="sum",
            inputs=[1, 2, 3],
            workers=["w1", "w2"],
        )
        assert prog.name.startswith("map_reduce_")
        assert len(prog.nodes) > 0

    def test_map_reduce_has_fork(self):
        prog = PatternLibrary.map_reduce(
            map_fn_name="f", reduce_fn_name="r",
            inputs=[1], workers=["w1"],
        )
        has_fork = any(isinstance(n, Fork) for n in prog.nodes)
        assert has_fork

    def test_map_reduce_has_merge(self):
        prog = PatternLibrary.map_reduce(
            map_fn_name="f", reduce_fn_name="r",
            inputs=[1, 2], workers=["w1"],
        )
        has_merge = any(isinstance(n, Merge) for n in prog.nodes)
        assert has_merge

    def test_map_reduce_merge_strategy_is_sum(self):
        prog = PatternLibrary.map_reduce(
            map_fn_name="f", reduce_fn_name="r",
            inputs=[1, 2], workers=["w1"],
        )
        merge_node = next(n for n in prog.nodes if isinstance(n, Merge))
        assert merge_node.strategy == MergeStrategy.SUM

    def test_map_reduce_empty_inputs(self):
        prog = PatternLibrary.map_reduce(
            map_fn_name="f", reduce_fn_name="r",
            inputs=[], workers=["w1"],
        )
        # Should compile without error
        gen = CodeGenerator()
        instrs = gen.generate(prog)
        assert instrs[-1].opcode == SupportOpcodes.HALT


class TestPatternScatterGather:
    """Tests for the scatter_gather pattern."""

    def test_scatter_gather_basic(self):
        prog = PatternLibrary.scatter_gather(
            task_name="analyze",
            task_payload="data",
            agents=["a1", "a2", "a3"],
        )
        assert prog.name.startswith("scatter_gather_")

    def test_scatter_gather_has_broadcast(self):
        prog = PatternLibrary.scatter_gather("t", "d", ["a1"])
        has_bcast = any(isinstance(n, Broadcast) for n in prog.nodes)
        assert has_bcast

    def test_scatter_gather_has_awaits(self):
        prog = PatternLibrary.scatter_gather("t", "d", ["a1", "a2"])
        awaits = [n for n in prog.nodes if isinstance(n, Await)]
        assert len(awaits) == 2

    def test_scatter_gather_has_concat_merge(self):
        prog = PatternLibrary.scatter_gather("t", "d", ["a1"])
        merge_node = next(n for n in prog.nodes if isinstance(n, Merge))
        assert merge_node.strategy == MergeStrategy.CONCAT


class TestPatternConsensus:
    """Tests for the consensus_round pattern."""

    def test_consensus_basic(self):
        prog = PatternLibrary.consensus_round(
            proposal="Do we deploy?",
            voters=["v1", "v2", "v3"],
        )
        assert prog.name.startswith("consensus_")

    def test_consensus_with_weights(self):
        prog = PatternLibrary.consensus_round(
            proposal="deploy",
            voters=["v1", "v2"],
            weights={"v1": 0.9, "v2": 0.3},
        )
        trusts = [n for n in prog.nodes if isinstance(n, Trust)]
        assert len(trusts) == 2

    def test_consensus_without_weights(self):
        prog = PatternLibrary.consensus_round(
            proposal="deploy",
            voters=["v1"],
        )
        trusts = [n for n in prog.nodes if isinstance(n, Trust)]
        assert len(trusts) == 0

    def test_consensus_has_vote_merge(self):
        prog = PatternLibrary.consensus_round(
            proposal="deploy",
            voters=["v1", "v2"],
        )
        merge_node = next(n for n in prog.nodes if isinstance(n, Merge))
        assert merge_node.strategy == MergeStrategy.VOTE


class TestPatternPipeline:
    """Tests for the pipeline pattern."""

    def test_pipeline_basic(self):
        prog = PatternLibrary.pipeline(
            stages=["outline", "draft", "edit"],
            agent_per_stage=["writer", "writer", "editor"],
        )
        assert prog.name == "pipeline"

    def test_pipeline_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            PatternLibrary.pipeline(
                stages=["a", "b"],
                agent_per_stage=["agent1"],
            )

    def test_pipeline_has_fork(self):
        prog = PatternLibrary.pipeline(["a"], ["agent1"])
        has_fork = any(isinstance(n, Fork) for n in prog.nodes)
        assert has_fork

    def test_pipeline_has_signals(self):
        prog = PatternLibrary.pipeline(["s1", "s2"], ["a1", "a2"])
        signals = [n for n in prog.nodes if isinstance(n, Signal)]
        assert len(signals) == 2

    def test_pipeline_has_join(self):
        prog = PatternLibrary.pipeline(["s1"], ["a1"])
        has_join = any(isinstance(n, Join) for n in prog.nodes)
        assert has_join


class TestPatternRetry:
    """Tests for the retry_with_backoff pattern."""

    def test_retry_basic(self):
        prog = PatternLibrary.retry_with_backoff(
            task_name="send_email",
            max_retries=3,
            base_delay=1.0,
        )
        assert prog.name == "retry_send_email"

    def test_retry_zero_retries(self):
        prog = PatternLibrary.retry_with_backoff("t", 0, 1.0)
        delegates = [n for n in prog.nodes if isinstance(n, Delegate)]
        assert len(delegates) == 0

    def test_retry_generates_reports(self):
        prog = PatternLibrary.retry_with_backoff("t", 3, 1.0)
        reports = [n for n in prog.nodes if isinstance(n, Report)]
        assert len(reports) == 3

    def test_retry_generates_delegates(self):
        prog = PatternLibrary.retry_with_backoff("t", 2, 1.0)
        delegates = [n for n in prog.nodes if isinstance(n, Delegate)]
        assert len(delegates) == 2

    def test_retry_has_signals(self):
        prog = PatternLibrary.retry_with_backoff("t", 3, 1.0)
        signals = [n for n in prog.nodes if isinstance(n, Signal)]
        # Signals for retry schedules (n-1, since last attempt has none)
        assert len(signals) == 2

    def test_retry_final_merge(self):
        prog = PatternLibrary.retry_with_backoff("t", 3, 1.0)
        merge_node = next(n for n in prog.nodes if isinstance(n, Merge))
        assert merge_node.strategy == MergeStrategy.MAX


# =========================================================================
# 10. High-level compile_coop_program
# =========================================================================

class TestCompileCoopProgram:
    """Tests for the compile_coop_program entry point."""

    def test_compiles_empty_program(self):
        prog = CoopProgram("empty")
        instrs = compile_coop_program(prog)
        assert instrs[-1].opcode == SupportOpcodes.HALT

    def test_compiles_with_optimization(self):
        prog = CoopProgram("test", [FluxInstruction.nop()])  # type: ignore
        # Can't add raw instruction; use a node instead
        prog = CoopProgram("test", [
            Task("t1", "a"),
            Task("t2", "b"),
        ])
        instrs = compile_coop_program(prog, optimize=True)
        assert instrs[-1].opcode == SupportOpcodes.HALT

    def test_compiles_without_optimization(self):
        prog = CoopProgram("test", [
            Task("t1", "alpha"),
        ])
        instrs = compile_coop_program(prog, optimize=False)
        assert any(i.opcode == A2A_Opcodes.DELEG for i in instrs)

    def test_compile_pattern(self):
        prog = PatternLibrary.map_reduce("f", "r", [1, 2], ["w1"])
        instrs = compile_coop_program(prog)
        assert instrs[-1].opcode == SupportOpcodes.HALT
        assert len(instrs) > 0


# =========================================================================
# 11. Integration tests
# =========================================================================

class TestIntegration:
    """End-to-end integration tests combining multiple compiler components."""

    def test_divide_conquer_synthesize_full(self):
        """Simulate a full divide-conquer-synthesize workflow."""
        prog = CoopProgram("dcs_full")
        # Phase 1: Divide
        subtasks = [Task(f"sub_{i}", f"worker_{i}") for i in range(3)]
        prog.add(Divide(task=Task("main", "owner"), subtasks=subtasks))
        # Phase 2: Set trust
        for w in ["worker_0", "worker_1", "worker_2"]:
            prog.add(Trust(agent=w, level=0.8))
        # Phase 3: Report start
        prog.add(Report(status="executing", data="DCS started"))
        # Phase 4: Merge
        prog.add(Merge(
            results=[f"result_{i}" for i in range(3)],
            strategy=MergeStrategy.VOTE,
        ))
        # Phase 5: Report done
        prog.add(Report(status="complete"))

        instrs = compile_coop_program(prog)
        assert instrs[-1].opcode == SupportOpcodes.HALT
        assert any(i.opcode == A2A_Opcodes.FORK for i in instrs)
        assert any(i.opcode == A2A_Opcodes.JOIN for i in instrs)
        assert any(i.opcode == A2A_Opcodes.TRUST for i in instrs)
        assert any(i.opcode == A2A_Opcodes.REPORT for i in instrs)
        assert any(i.opcode == A2A_Opcodes.MERGE for i in instrs)

    def test_map_reduce_compiles_to_valid_bytecode(self):
        """Map-reduce pattern should compile to valid bytecode."""
        prog = PatternLibrary.map_reduce("analyze", "sum", [1, 2, 3, 4], ["w1", "w2"])
        instrs = compile_coop_program(prog)
        # Every instruction should have a valid opcode
        for instr in instrs:
            assert isinstance(instr.opcode, int)
            assert 0x00 <= instr.opcode <= 0xFF

    def test_pipeline_compiles_to_valid_bytecode(self):
        """Pipeline pattern should compile to valid bytecode."""
        prog = PatternLibrary.pipeline(
            stages=["parse", "analyze", "render"],
            agent_per_stage=["parser", "analyzer", "renderer"],
        )
        instrs = compile_coop_program(prog)
        for instr in instrs:
            assert 0x00 <= instr.opcode <= 0xFF

    def test_retry_compiles_to_valid_bytecode(self):
        """Retry pattern should compile to valid bytecode."""
        prog = PatternLibrary.retry_with_backoff("api_call", 5, 0.5, "http_agent")
        instrs = compile_coop_program(prog)
        for instr in instrs:
            assert 0x00 <= instr.opcode <= 0xFF

    def test_consensus_compiles_to_valid_bytecode(self):
        """Consensus pattern should compile to valid bytecode."""
        prog = PatternLibrary.consensus_round(
            proposal="Deploy v2?",
            voters=["alice", "bob", "carol"],
            weights={"alice": 0.9, "bob": 0.7, "carol": 0.5},
        )
        instrs = compile_coop_program(prog)
        for instr in instrs:
            assert 0x00 <= instr.opcode <= 0xFF

    def test_multi_agent_communication(self):
        """Test a program with complex multi-agent communication."""
        prog = CoopProgram("multi_comm")
        # Tell data to agent
        prog.add(Tell(agent="alpha", data="context"))
        # Ask agent for analysis
        prog.add(Ask(agent="alpha", query="analyze this"))
        # Delegate work
        prog.add(Delegate(task=Task("compute", "alpha"), target_agent="alpha"))
        # Set trust
        prog.add(Trust(agent="alpha", level=0.95))
        # Report
        prog.add(Report(status="analysis_complete"))
        # Signal
        prog.add(Signal(name="analysis_done"))
        # Await response
        prog.add(Await(signal_name="acknowledged"))

        instrs = compile_coop_program(prog)
        a2a_opcodes = {A2A_Opcodes.TELL, A2A_Opcodes.ASK, A2A_Opcodes.DELEG,
                       A2A_Opcodes.TRUST, A2A_Opcodes.REPORT,
                       A2A_Opcodes.SIGNAL, A2A_Opcodes.AWAIT}
        found_opcodes = {i.opcode for i in instrs}
        for op in a2a_opcodes:
            assert op in found_opcodes, f"Missing opcode {hex(op)}"

    def test_optimizer_on_compiled_program(self):
        """Optimizing a compiled program should produce valid bytecode."""
        prog = CoopProgram("opt_test")
        for i in range(5):
            prog.add(Task(f"task_{i}", "alpha"))
        instrs = compile_coop_program(prog, optimize=True)
        assert instrs[-1].opcode == SupportOpcodes.HALT
        assert len(instrs) > 0

    def test_serialize_all_instructions(self):
        """All compiled instructions should be serializable to bytes."""
        prog = CoopProgram("serialize_test")
        prog.add(Task("t", "a"))
        prog.add(Tell("a", "data"))
        prog.add(Ask("a", "query"))
        instrs = compile_coop_program(prog, optimize=False)
        for instr in instrs:
            b = instr.to_bytes()
            assert len(b) >= 1
            assert b[0] == instr.opcode

    def test_scatter_gather_with_many_agents(self):
        """Scatter-gather with many agents."""
        agents = [f"agent_{i}" for i in range(8)]
        prog = PatternLibrary.scatter_gather("task", "payload", agents)
        instrs = compile_coop_program(prog)
        assert instrs[-1].opcode == SupportOpcodes.HALT


# =========================================================================
# 12. Register constants
# =========================================================================

class TestRegisterConstants:
    """Tests for register allocation constants."""

    def test_scratch_is_zero(self):
        assert R_SCRATCH == 0

    def test_task_register_range(self):
        assert R_TASK_START == 1
        assert R_TASK_END == 7

    def test_agent_register_range(self):
        assert R_AGENT_START == 8
        assert R_AGENT_END == 15

    def test_task_register_count(self):
        assert R_TASK_END - R_TASK_START + 1 == 7

    def test_agent_register_count(self):
        assert R_AGENT_END - R_AGENT_START + 1 == 8
