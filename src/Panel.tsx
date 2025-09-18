import { Button } from "@fiftyone/components";
import {
  useOperatorExecutor,
  Operator,
  OperatorConfig,
  registerOperator,
} from "@fiftyone/operators";
import * as fos from "@fiftyone/state";
import { State } from "@fiftyone/state";
import { useEffect, useState } from "react";
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
  LinearProgress,
  Alert,
  Checkbox,
} from "@mui/material";

type ProgressInfo = {
  progress: number;
  label: string;
  operation?: string;
};

type ActiveOperations = {
  [key: string]: ProgressInfo;
};

export function Panel() {
  // Debug mode
  const [debugMode, setDebugMode] = useState(true);

  // Global FiftyOne state
  const dataset: State.Dataset | null = useRecoilValue(fos.dataset);
  const filters = useRecoilValue(fos.filters);

  // Local plugin state
  const [annotations, setAnnotations] = useState<any[]>([]);
  const [doProgressPoll, setDoProgressPoll] = useState(false);
  const [activeOperations, setActiveOperations] = useState<ActiveOperations>({});
  const [clusteringResults, setClusteringResults] = useState<any>(null);

  // Python operator: compute embeddings
  const computeEmbeddingsExecutor = useOperatorExecutor(
    "@bifrostai/cvcov/compute_embeddings",
    {
      onSuccess: (result: any) => {
        console.log("Embedding computation completed:", result);
        // Auto-continue with clustering after embeddings complete
        clusterEmbeddingsExecutor.execute({});
      },
      onError: (error: any) => {
        console.log("Embedding computation failed:", error);
        setDoProgressPoll(false);
      },
    }
  ) as any;

  // Python operator: cluster embeddings
  const clusterEmbeddingsExecutor = useOperatorExecutor(
    "@bifrostai/cvcov/cluster_embeddings",
    {
      onSuccess: (result: any) => {
        console.log("Clustering completed:", result);
        setDoProgressPoll(false);
        setActiveOperations({});
        // Extract the actual result data from the nested structure
        setClusteringResults(result.result);
      },
      onError: (error: any) => {
        console.log("Clustering failed:", error);
        setDoProgressPoll(false);
      },
    }
  ) as any;

  // Python operator: fetch annotations
  const fetchAnnotationsExecutor = useOperatorExecutor(
    "@bifrostai/cvcov/fetch_annotations"
  );

  // Python operator: get progress
  const getProgressExecutor = useOperatorExecutor(
    "@bifrostai/cvcov/get_progress"
  );

  useEffect(() => {
    if (dataset) {
      fetchAnnotationsExecutor.execute({});
    }
  }, [dataset, filters]);

  useEffect(() => {
    if (fetchAnnotationsExecutor.result?.annotations) {
      setAnnotations(fetchAnnotationsExecutor.result.annotations);
    }
  }, [fetchAnnotationsExecutor.result]);

  // Poll for progress updates using operator
  useEffect(() => {
    if (!doProgressPoll) return;

    const interval = setInterval(() => {
      getProgressExecutor.execute({}); // No operation_name = get all active operations
    }, 500); // Poll every 500ms

    return () => clearInterval(interval);
  }, [doProgressPoll]);

  // Handle progress updates for all active operations
  useEffect(() => {
    if (getProgressExecutor.result) {
      const result = getProgressExecutor.result;

      if (result.active_operations) {
        const operations = result.active_operations;
        const newActiveOperations: ActiveOperations = {};

        // Process all active operations
        for (const [operationName, progressData] of Object.entries(operations)) {
          newActiveOperations[operationName] = {
            progress: (progressData as any).progress,
            label: (progressData as any).label,
            operation: operationName,
          };
        }

        setActiveOperations(newActiveOperations);

        // Stop polling if no active operations or all completed
        const hasActiveOperations = Object.values(newActiveOperations).some(
          (op) => op.progress < 1
        );
        if (!hasActiveOperations) {
          setDoProgressPoll(false);
        }
      } else if (result.progress !== undefined) {
        // Handle single operation response (fallback)
        setActiveOperations({
          unknown: {
            progress: result.progress,
            label: result.label,
            operation: "unknown",
          },
        });

        if (result.progress >= 1) {
          setDoProgressPoll(false);
        }
      } else {
        // No operations found
        setActiveOperations({});
        setDoProgressPoll(false);
      }
    }
  }, [getProgressExecutor.result]);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Bifrost CVCov Plugin
      </Typography>
      <Typography variant="body1" gutterBottom>
        Dataset: {dataset?.name || "No dataset loaded"}
      </Typography>

      <Box sx={{ mt: 2, mb: 2 }}>
        <Checkbox
          checked={debugMode}
          onChange={() => {
            setDebugMode(!debugMode);
          }}
        />
        <Typography variant="body2" display="inline" sx={{ ml: 1 }}>
          Debug mode
        </Typography>
      </Box>

      <Button
        variant="contained"
        disabled={doProgressPoll}
        onClick={() => {
          setDoProgressPoll(true);
          setActiveOperations({});
          computeEmbeddingsExecutor.execute({});
        }}
      >
        {doProgressPoll ? "Processing..." : "Compute Embeddings & Cluster"}
      </Button>

      {/* Show progress bars for all active operations */}
      {Object.keys(activeOperations).length > 0 && (
        <Box sx={{ mt: 2, mb: 2 }}>
          {Object.entries(activeOperations).map(([operationName, progressInfo]) => (
            <Box key={operationName} sx={{ mb: 1 }}>
              <Typography variant="body2" gutterBottom>
                {operationName.charAt(0).toUpperCase() + operationName.slice(1)}: {progressInfo.label} ({Math.round(progressInfo.progress * 100)}% complete)
              </Typography>
              <LinearProgress
                variant="determinate"
                value={progressInfo.progress * 100}
              />
            </Box>
          ))}
        </Box>
      )}

      {/* Show completion message */}
      {computeEmbeddingsExecutor.result?.message &&
        !computeEmbeddingsExecutor.isExecuting && (
          <Alert severity="success" sx={{ mt: 2, mb: 2 }}>
            {computeEmbeddingsExecutor.result.message}
          </Alert>
        )}

      {/* Show clustering results */}
      {clusteringResults && (
        <Box sx={{ mt: 2, mb: 2 }}>
          <Alert severity="success" sx={{ mb: 2 }}>
            {clusteringResults.message}
          </Alert>

          <Typography variant="h6" gutterBottom>
            Clustering Results
          </Typography>

          <Box sx={{ display: "flex", gap: 4, flexWrap: "wrap", mb: 2 }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Number of Clusters
              </Typography>
              <Typography variant="h4" color="primary">
                {clusteringResults.num_clusters}
              </Typography>
            </Box>

            <Box>
              <Typography variant="body2" color="text.secondary">
                Total Objects
              </Typography>
              <Typography variant="h4">
                {clusteringResults.labels?.length || 0}
              </Typography>
            </Box>

            {clusteringResults.labels && (
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Noise Points (Cluster -1)
                </Typography>
                <Typography variant="h4" color="error">
                  {clusteringResults.labels.filter((label: number) => label === -1).length}
                </Typography>
              </Box>
            )}
          </Box>

          {clusteringResults.labels && (
            <Box>
              <Typography variant="body2" gutterBottom>
                Cluster Distribution:
              </Typography>
              <Box sx={{ maxHeight: 200, overflow: "auto" }}>
                {Object.entries(
                  clusteringResults.labels.reduce((acc: any, label: number) => {
                    acc[label] = (acc[label] || 0) + 1;
                    return acc;
                  }, {})
                )
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([cluster, count]) => (
                    <Box key={cluster} sx={{ display: "flex", justifyContent: "space-between", py: 0.5 }}>
                      <Typography variant="body2">
                        Cluster {cluster === "-1" ? "(Noise)" : cluster}:
                      </Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {count} objects
                      </Typography>
                    </Box>
                  ))}
              </Box>
            </Box>
          )}
        </Box>
      )}

      {/* Debug info: show annotations table */}
      {debugMode && (
        <>
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
        </>
      )}
    </Box>
  );
}

// Example frontend operator
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
registerOperator(AlertOperator, "@bifrostai/cvcov");
