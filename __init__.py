import fiftyone.operators as foo
from fiftyone.operators.executor import ExecutionContext


class MyPythonOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="my_python_operator",
            label="My Python Operator",
        )

    def execute(self, ctx: ExecutionContext):
        ctx.log("Python operator executing!")
        available = False
        try:
            import torch
            available = torch.cuda.is_available()
        except ImportError:
            ctx.log("PyTorch is not installed.")
        return {
            "pytorch_available": available,
            "message": "Hello from Python!",
            "status": "success",
        }


class FetchAnnotationsOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="fetch_annotations",
            label="Fetch Annotations",
        )

    def execute(self, ctx: ExecutionContext):
        import fiftyone as fo

        dataset = ctx.dataset
        view = ctx.view if ctx.view else dataset

        annotations = []

        ctx.log(f"Processing {len(view)} samples for annotations...")

        for sample in view.iter_samples():
            sample_id = str(sample.id)

            # Iterate through all fields in the sample
            for field_name, field_value in sample.iter_fields():

                # Handle Detections
                if isinstance(field_value, fo.Detections):
                    for idx, detection in enumerate(field_value.detections):
                        annotations.append({
                            "id": f"{sample_id}-{field_name}-{idx}",
                            "sampleId": sample_id,
                            "type": "Detection",
                            "label": detection.label,
                            "confidence": round(detection.confidence, 3) if detection.confidence else "N/A",
                            "field": field_name
                        })

                # Handle Classifications
                elif isinstance(field_value, fo.Classification):
                    annotations.append({
                        "id": f"{sample_id}-{field_name}",
                        "sampleId": sample_id,
                        "type": "Classification",
                        "label": field_value.label,
                        "confidence": round(field_value.confidence, 3) if field_value.confidence else "N/A",
                        "field": field_name
                    })

        ctx.log(f"Found {len(annotations)} total annotations")

        return {
            "annotations": annotations,
            "total_count": len(annotations),
            "message": f"Found {len(annotations)} annotations"
        }


def register(plugin):
    plugin.register(MyPythonOperator)
    plugin.register(FetchAnnotationsOperator)
