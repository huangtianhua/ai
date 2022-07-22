import onnxruntime as ort
import numpy as np
from ctypes import c_long


def run(x: np.array) -> np.array:
    options = ort.SessionOptions()
    sess = ort.InferenceSession("../../models/debug.onnx", sess_options=options, providers=['CANNExecutionProvider', 'CPUExecutionProvider'])
    output = sess.run(["output"], {"input": x})
    print(c_long.from_address(id(sess)).value)
    return output[0]


def main():
    input = np.int64([1, 2, 3])
    print(run(x=input))
    print("----------")


if __name__ == "__main__":
    main()
