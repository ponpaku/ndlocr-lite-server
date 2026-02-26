"""Diagnose Reshape constants in PARSEQ ONNX model.

Finds ALL Reshape nodes and their shape inputs, distinguishing between
graph.initializer constants vs inline Constant op outputs.

Usage: python tools/diagnose_reshapes.py
"""
import sys
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh
from onnx import TensorProto

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "src" / "model"
MODEL = "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"


def diagnose(model_path: str) -> None:
    model = onnx.load(model_path)
    graph = model.graph

    # Build maps
    init_map = {i.name: i for i in graph.initializer}
    const_op_map = {}  # output_name -> Constant node
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            const_op_map[node.output[0]] = node

    print(f"graph.initializer count : {len(init_map)}")
    print(f"Constant op count       : {len(const_op_map)}")
    print()

    init_batch1 = 0
    const_batch1 = 0
    other = 0

    print("=== All Reshape nodes ===")
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_input = node.input[1]
        node_name = node.name or "(unnamed)"
        out_name = node.output[0] if node.output else "?"

        if shape_input in init_map:
            arr = nh.to_array(init_map[shape_input])
            src = "INIT"
            has_batch0 = (arr.ndim == 1 and len(arr) > 0 and arr[0] == 1)
            if has_batch0:
                init_batch1 += 1
            else:
                other += 1
            flag = " *** arr[0]==1 ***" if has_batch0 else ""
            print(f"  [{src}] {node_name!r}  shape_input={shape_input!r}  arr={arr.tolist()}{flag}")

        elif shape_input in const_op_map:
            cnode = const_op_map[shape_input]
            arr = None
            for attr in cnode.attribute:
                if attr.name == "value":
                    arr = nh.to_array(attr.t)
                    break
            src = "CONST_OP"
            if arr is not None:
                has_batch0 = (arr.ndim == 1 and len(arr) > 0 and arr[0] == 1)
                if has_batch0:
                    const_batch1 += 1
                else:
                    other += 1
                flag = " *** arr[0]==1 ***" if has_batch0 else ""
                print(f"  [{src}] {node_name!r}  shape_input={shape_input!r}  arr={arr.tolist()}{flag}")
            else:
                other += 1
                print(f"  [{src}] {node_name!r}  shape_input={shape_input!r}  (no value attr)")
        else:
            other += 1
            print(f"  [DYNAMIC] {node_name!r}  shape_input={shape_input!r}  (computed at runtime)")

    print()
    print(f"Initializer Reshape constants with arr[0]==1 : {init_batch1}")
    print(f"Constant op Reshape constants with arr[0]==1 : {const_batch1}")
    print(f"Other (dynamic / no-batch-at-0)              : {other}")
    print()

    # Specifically look for cross_attn
    print("=== Cross-attention Reshape nodes ===")
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        node_name = node.name or node.output[0] if node.output else "?"
        if "cross_attn" not in (node.name or "") and "cross_attn" not in (node.output[0] if node.output else ""):
            continue
        shape_input = node.input[1]
        if shape_input in init_map:
            arr = nh.to_array(init_map[shape_input])
            print(f"  [INIT]     {node.name!r}  out={node.output[0]!r}  shape={arr.tolist()}")
        elif shape_input in const_op_map:
            cnode = const_op_map[shape_input]
            arr = None
            for attr in cnode.attribute:
                if attr.name == "value":
                    arr = nh.to_array(attr.t)
                    break
            print(f"  [CONST_OP] {node.name!r}  out={node.output[0]!r}  shape={arr.tolist() if arr is not None else '?'}")
        else:
            print(f"  [DYNAMIC]  {node.name!r}  out={node.output[0]!r}  shape_input={shape_input!r}")


if __name__ == "__main__":
    model_path = str(MODEL_DIR / MODEL)
    print(f"Diagnosing: {model_path}")
    print()
    diagnose(model_path)
