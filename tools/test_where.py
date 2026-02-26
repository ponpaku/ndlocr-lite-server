"""Test which Where op configurations work in onnxruntime."""
import onnx, onnxruntime as ort, numpy as np, io, sys
from onnx import helper, TensorProto

def test_where(cond_shape, x_shape, y_shape, cond_dtype, x_dtype, label):
    cond = helper.make_tensor_value_info('cond', cond_dtype, cond_shape)
    x = helper.make_tensor_value_info('x', x_dtype, x_shape)
    y = helper.make_tensor_value_info('y', x_dtype, y_shape)
    out = helper.make_tensor_value_info('out', x_dtype, x_shape)
    node = helper.make_node('Where', inputs=['cond', 'x', 'y'], outputs=['out'])
    graph = helper.make_graph([node], 'test', [cond, x, y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 9)], ir_version=8)
    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)
    try:
        sess = ort.InferenceSession(buf.read(), providers=['CPUExecutionProvider'])
        return 'LOAD_OK'
    except Exception as e:
        return f'FAIL: {str(e)[:100]}'

print('ort version:', ort.__version__)
print()
print('bool_1d + float_1d:', test_where([None], [None], [None], TensorProto.BOOL, TensorProto.FLOAT, ''))
print('bool_2d + float_2d:', test_where([None, None], [None, None], [None, None], TensorProto.BOOL, TensorProto.FLOAT, ''))
print('bool_2d + int64_2d:', test_where([None, None], [None, None], [None, None], TensorProto.BOOL, TensorProto.INT64, ''))
print('bool_3d + float_3d:', test_where([None,None,None], [None,None,None], [None,None,None], TensorProto.BOOL, TensorProto.FLOAT, ''))
print('bool_4d + float_4d:', test_where([None,None,None,None], [None,None,None,None], [None,None,None,None], TensorProto.BOOL, TensorProto.FLOAT, ''))

# Also check original ONNX model's Where node
m = onnx.load('src/model/parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx')
for n in m.graph.node:
    if n.op_type == 'Where':
        print(f'\nOriginal model Where node: {n.name}')
        print(f'  inputs: {list(n.input)}')
        # Find value_info for inputs
        vi_map = {vi.name: vi for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
        for inp in n.input:
            if inp in vi_map:
                t = vi_map[inp].type.tensor_type
                dtype = t.elem_type
                shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in t.shape.dim]
                print(f'    {inp}: dtype={dtype} shape={shape}')
