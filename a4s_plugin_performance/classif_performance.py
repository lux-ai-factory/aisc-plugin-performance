# src/my_a4s_plugin/plugin.py
import csv
import io
from a4s_plugin_interface.models.measure import Measure
from a4s_plugin_interface import metric
from a4s_plugin_interface.base_evaluation_plugin import BaseEvaluationPlugin
from a4s_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from a4s_plugin_interface.input_providers.csv_input_provider import CsvInputProvider
from pydantic import BaseModel


METRIC_ACCURACY = "Accuracy"
METRIC_PRECISION = "Precision"
METRIC_RECALL = "Recall"
METRIC_F1_SCORE = "F1-Score"
METRIC_MCC = "MCC"

class ConfigForSchema(BaseModel):
    pass


class ParquetInputProvider(BaseInputProvider):
    """
    A concrete implementation of BaseInputProvider for parquet files.
    Parses the file content into a list of dictionaries, where each dict represents a row.
    """
    import pandas as pd

    def _read_data(self, file_content) -> pd.DataFrame:
        """
        Converts parquet bytes into a list of .
        """
        import pandas as pd
        
        file_stream = io.BytesIO(file_content)
        file_stream.seek(0)
        return pd.read_parquet(file_stream)


class OnnxInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes):
        from onnxruntime import InferenceSession

        session: InferenceSession = InferenceSession(file_content)
        return session

class ClassificationPerformancePlugin(BaseEvaluationPlugin[ConfigForSchema]):

    def set_dataset_input_provider(self, file_content: bytes | None) -> BaseInputProvider:
        self.dataset_input_provider = ParquetInputProvider(file_content)
        return self.dataset_input_provider


    def set_model_input_provider(self, file_content: bytes | None) -> BaseInputProvider:
        self.model_input_provider = OnnxInputProvider(file_content)
        return self.model_input_provider


    def _get_y_pred_probs(self, session, x_test_np):
        import numpy as np
        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[1].name
        pred_onx = session.run([label_name], {input_name: x_test_np})[0]
        y_pred_proba = np.array([list(d.values()) for d in pred_onx])
        return y_pred_proba

    def evaluate(self, config_data: dict) -> list[Measure]:
        from onnxruntime import InferenceSession
        import pandas as pd
        import numpy as np
        
        # config: ConfigForSchema = self.validate_config_form_data(config_data)

        df_test: pd.DataFrame = self.get_dataset()
        session: InferenceSession = self.get_model()

        # TODO: We need to find a way to handle datashape?
        # More specifically, the input features for the model?
        # Thinking: use config_data to pass with target feature name and date feature name?
        x_test_np = df_test[
            [f for f in df_test.columns if f not in ["issue_d", "charged_off"]]
        ].to_numpy()

        y_true = df_test["charged_off"].to_numpy()

        y_pred_proba = self._get_y_pred_probs(session, x_test_np)
        y_pred = np.argmax(y_pred_proba, axis=1)


        return {
            "y_true": y_true, 
            "y_pred": y_pred
        }


    @metric(METRIC_ACCURACY)
    def metric_accuracy(self, results: dict)-> list[Measure]:
        from sklearn.metrics import accuracy_score
        return [
            Measure(
                name=METRIC_ACCURACY, 
                score=accuracy_score(results["y_true"], results["y_pred"])
            )
        ]
    

    @metric(METRIC_F1_SCORE)
    def metric_f1(self, results: dict) -> list[Measure]:
        from sklearn.metrics import f1_score
        f1 = f1_score(results["y_true"], results["y_pred"], zero_division=0)
        return [
            Measure(
                name=METRIC_F1_SCORE, 
                score=f1
            )
        ]
    
    @metric(METRIC_PRECISION)
    def metric_precision(self, results: dict) -> list[Measure]:
        from sklearn.metrics import precision_score
        precision = precision_score(results["y_true"], results["y_pred"], zero_division=0)
        return [Measure(name=METRIC_PRECISION, score=precision)]


    @metric(METRIC_RECALL)
    def metric_recall(self, results: dict) -> list[Measure]:
        from sklearn.metrics import recall_score
        recall = recall_score(results["y_true"], results["y_pred"], zero_division=0)
        return [Measure(name=METRIC_RECALL, score=recall)]
    

    @metric(METRIC_MCC)
    def metric_mcc(self, results: dict) -> list[Measure]:
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(results["y_true"], results["y_pred"])
        return [Measure(name=METRIC_MCC, score=mcc)]
    

    @metric("Confusion_Matrix")
    def metric_confusion_matrix(self, results: dict) -> list[Measure]:
        from sklearn.metrics import confusion_matrix
        metrics: list[Measure] = []

        matrix = (confusion_matrix(results["y_true"], results["y_pred"]),)
        matrix = matrix[0]
        max_i, max_j = matrix.shape

        print(matrix)

        for i in range(max_i):
            for j in range(max_j):
                metric = Measure(
                    name="Confusion Matrix",
                    description=f"({i},{j})/({max_i},{max_j})",
                    score=matrix[i][j],
                )
                metrics.append(metric)

        return metrics