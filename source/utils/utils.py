import onnxruntime as ort
import onnx

class WrapInferenceSession:

    def __init__(self, onnx_path):
        onnx_bytes = onnx.load(onnx_path)
        self.sess = ort.InferenceSession(onnx_bytes.SerializeToString())
        self.onnx_bytes = onnx_bytes

    def run(self, *args):
        return self.sess.run(*args)

    def get_inputs(self, *args):
        return self.sess.get_inputs(*args)

    def get_outputs(self, *args):
        return self.sess.get_outputs(*args)

    def __getstate__(self):
        return {'onnx_bytes': self.onnx_bytes}

    def __setstate__(self, values):
        onnx_bytes = values['onnx_bytes']
        self.sess = ort.InferenceSession(onnx_bytes.SerializeToString())