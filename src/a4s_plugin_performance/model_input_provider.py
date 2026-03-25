from a4s_plugin_interface.input_providers.base_input_provider import BaseInputProvider


class OnnxInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes):
        from onnxruntime import InferenceSession

        session: InferenceSession = InferenceSession(file_content)
        return session

    def predict(self, x_test, probabilities=False):
        import pandas as pd
        import numpy as np

        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.to_numpy()

        if not isinstance(x_test, np.ndarray):
            raise ValueError(f"x_test should be np.ndarray, found: {type(x_test)}")

        if self._data is None:
            raise ValueError(
                f"Please call {self.__class__.__name__}._read_data(...) first"
            )

        session = self._data

        # determine expected dtype
        expected = session.get_inputs()[0].type
        dtype = (
            np.float32
            if "tensor(float)" in expected
            else (np.float64 if "tensor(double)" in expected else x_test.dtype)
        )

        x = np.ascontiguousarray(x_test.astype(dtype, copy=False))
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Get output candidates
        out_candidates = session.get_outputs()

        if probabilities:
            # prefer a 'prob'/'probab' output if available (for classification)
            idx = next(
                (i for i, o in enumerate(out_candidates) if "prob" in o.name.lower()),
                None,
            )
            if idx is None:
                # fallback: second output if it exists, else first
                idx = int(len(out_candidates) >= 2)
        else:
            # for regression, use first output
            idx = 0
        output_name = out_candidates[idx].name

        raw = session.run([output_name], {session.get_inputs()[0].name: x})[0]

        # handle ZipMap case for classification probabilities (list of dicts)
        if probabilities and isinstance(raw, list) and raw and isinstance(raw[0], dict):
            keys = list(raw[0].keys())
            arr = np.array(
                [[row.get(k, 0.0) for k in keys] for row in raw], dtype=np.float32
            )
            return arr

        arr = np.array(raw)
        if probabilities and arr.ndim == 1:
            arr = arr[:, None]
        if not probabilities:
            arr = arr.squeeze(-1)

        return arr.astype(np.float32, copy=False)
