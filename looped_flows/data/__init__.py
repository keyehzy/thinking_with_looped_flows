"""Benchmark tasks. ``build_task`` constructs a task from a config dict."""

from __future__ import annotations

from .common import Task, TaskSpec, batch_to


def build_task(name: str, **kwargs) -> Task:
    if name == "sudoku":
        from .sudoku import SudokuTask

        return SudokuTask(**kwargs)
    if name == "maze":
        from .maze import MazeTask

        return MazeTask(**kwargs)
    if name == "nqueens":
        from .nqueens import NQueensTask

        return NQueensTask(**kwargs)
    if name == "graph_coloring":
        from .graph_coloring import GraphColoringTask

        return GraphColoringTask(**kwargs)
    if name == "arc":
        from .arc import ARCTask

        return ARCTask(**kwargs)
    raise ValueError(f"unknown task {name}")


__all__ = ["Task", "TaskSpec", "batch_to", "build_task"]
