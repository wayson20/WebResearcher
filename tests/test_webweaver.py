# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for WebWeaver Agent
"""
import pytest
import asyncio
import os
import sys

sys.path.append("..")
from webresearcher.tool_memory import MemoryBank, RetrieveTool
from webresearcher.tool_planner_search import PlannerSearchTool
from webresearcher.web_weaver_agent import (
    WebWeaverAgent,
    WebWeaverPlanner,
    WebWeaverWriter,
)


def test_memory_bank_basic():
    """Test basic MemoryBank operations."""
    memory = MemoryBank()

    # Test add_evidence
    result = memory.add_evidence(
        content="Full evidence content here",
        summary="Short summary"
    )

    assert "id_1" in result
    assert "Short summary" in result
    assert memory.size() == 1

    # Test retrieve
    retrieved = memory.retrieve(["id_1"])
    assert "Full evidence content here" in retrieved
    assert "id='id_1'" in retrieved

    # Test retrieve non-existent ID
    retrieved_error = memory.retrieve(["id_999"])
    assert "not found" in retrieved_error

    # Test get_all_ids
    ids = memory.get_all_ids()
    assert ids == ["id_1"]

    # Test clear
    memory.clear()
    assert memory.size() == 0


def test_retrieve_tool():
    """Test RetrieveTool."""
    memory = MemoryBank()
    memory.add_evidence("Content 1", "Summary 1")
    memory.add_evidence("Content 2", "Summary 2")

    tool = RetrieveTool(memory)

    # Test tool properties
    assert tool.name == "retrieve"
    assert "citation_ids" in tool.parameters["properties"]

    # Test call method
    result = tool.call({"citation_ids": ["id_1", "id_2"]})
    assert "Content 1" in result
    assert "Content 2" in result


if __name__ == "__main__":
    test_memory_bank_basic()

    test_retrieve_tool()
