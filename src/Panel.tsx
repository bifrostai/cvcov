import { Button } from "@fiftyone/components";
import {
  executeOperator,
  useOperatorExecutor,
  Operator,
  OperatorConfig,
  registerOperator,
} from "@fiftyone/operators";
import * as fos from "@fiftyone/state";
import { State } from "@fiftyone/state";
import { useCallback } from "react";
import { useRecoilValue } from "recoil";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
} from "@mui/material";

import { useEffect, useState } from "react";

export function Panel() {
  const dataset: State.Dataset | null = useRecoilValue(fos.dataset);
  const viewStages: State.Stage[] | null = useRecoilValue(fos.view);
  const [annotations, setAnnotations] = useState<any[]>([]);

  // Use Python operator to get the actual data since React atoms don't expose sample content
  const fetchAnnotationsExecutor = useOperatorExecutor(
    "@bifrostai/cvcov/fetch_annotations"
  );

  useEffect(() => {
    if (dataset) {
      console.log("Fetching annotations for dataset:", dataset.name);
      fetchAnnotationsExecutor.execute({});
    }
  }, [dataset, viewStages]);

  useEffect(() => {
    if (fetchAnnotationsExecutor.result?.annotations) {
      console.log(
        "Got annotations:",
        fetchAnnotationsExecutor.result.annotations
      );
      setAnnotations(fetchAnnotationsExecutor.result.annotations);
    }
  }, [fetchAnnotationsExecutor.result]);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Bifrost CVCov Plugin
      </Typography>
      <Typography variant="body1" gutterBottom>
        Dataset: {dataset?.name || "No dataset loaded"}
      </Typography>

      <Typography variant="body2" gutterBottom>
        Samples in view: {fetchAnnotationsExecutor.result?.total_count || 0}
      </Typography>

      <Typography variant="h6" sx={{ mt: 2, mb: 2 }}>
        Annotated Objects ({annotations.length})
      </Typography>

      <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>Sample ID</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Label</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Field</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {annotations.map((annotation) => (
              <TableRow key={annotation.id}>
                <TableCell>{annotation.sampleId}</TableCell>
                <TableCell>{annotation.type}</TableCell>
                <TableCell>{annotation.label}</TableCell>
                <TableCell>{annotation.confidence}</TableCell>
                <TableCell>{annotation.field}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {annotations.length === 0 && !fetchAnnotationsExecutor.isLoading && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          No annotations found in the current dataset view.
        </Typography>
      )}
    </Box>
  );
}

// Example of a frontend operator that shows an alert
class AlertOperator extends Operator {
  get config() {
    return new OperatorConfig({
      name: "show_alert",
      label: "Show alert",
      unlisted: true,
    });
  }
  async execute() {
    alert(`Hello from plugin ${this.pluginName}`);
  }
}

// Register the operators to the plugin
registerOperator(AlertOperator, "@bifrostai/cvcov");
