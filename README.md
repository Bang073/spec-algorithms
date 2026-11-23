# Spec Algorithms

Spec Algorithms is a Python package for retrieving behavioral and safety specifications, generating and scoring guided answers, and measuring how a change to a graph path affects an answer. It contains the algorithms, a local command-line runner and offline tests. To run inference, supply your own records, graph snapshot and model weights.

## Algorithms

| Command | Operation | Result |
| --- | --- | --- |
| `alignment-value` | `retrieve` | Ranks applicable L2 rules through L3 value routes. Returns rule IDs, scores and the graph hash. |
| `alignment-value` | `align` | Generates an answer using retrieved rules and evidence. With a matching evaluation case, also returns a per-rule judgment against its fixed specification. |
| `alignment-value` | `represent` | Returns four values for each selected training record. S measures scarcity in the graph, C measures preservation of applicable cultural evidence, E measures similarity to ethical anchors, and T records the outcome of a scored training answer. |
| `path-influence` | `remove_content`, `remove_edge`, `rewire_edge`, `restore` | Compares a baseline with four interventions across paired risk and benign questions. Returns changes in safety and helpfulness, pair counts and restoration checks. |

A pending T or an inapplicable C is reported as `null`. Path influence is measured at inference time. Its `complete` field follows the benchmark threshold of at least 100 executed pairs and a score for every executed run. Use `paired_cases_executed`, `runs` and `scored` to inspect a smaller run.

## Install and check the package

From the repository root, run the following commands in a Linux or macOS shell with Python 3.12 or later. The CLI help and offline tests require only the base package and its dependencies.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
specalgo --help
python -m unittest discover -s tests -v
```

For CLI inference, the bundled adapters require a compatible runtime and local snapshots of the target, judge and embedding models. Set `HF_HOME` if you use a custom Hugging Face cache. To use a different runtime through the Python API, pass compatible `model` and `embedder` adapters to `SpecAlignmentValueAlgorithm`. [Default model adapters](src/specalgo/models.py)

## Provide local inputs

Pass the read-only resource directory with `--resources` and a separate writable directory with `--work-dir`. The defaults are `./resources` and `./runs`. The equivalent environment variables are `SPECALGO_RESOURCES` and `SPECALGO_WORK_DIR`.

| Relative path under the resource directory | Used by |
| --- | --- |
| `data/train.jsonl` | All operations. One `Record` per line with `case_id`, `spec_id`, `original_data`, `spec`, `values` and `provenance`. |
| `data/splits/manifest.json` | All operations. The training split uses `train_count` and `train_hash`. Evaluation records use `test_count` and `test_hash`. |
| `artifacts/graph/base.json` | All operations. A typed graph with `meta`, `nodes` and `edges`. Its `meta.train_hash` must match the input records. |
| `data/test_private/records.jsonl` and `data/test_private/cases.jsonl` | Scored alignment and path interventions. Cases link a `test_id`, `case_id`, `query_type` and `query` to a source record. Interventions require a risk and a benign question for each selected case. |
| `artifacts/train_observations/results.jsonl` | Optional input for T in `represent`. Supply previously scored risk observations from the graph-guided phase. Use `--observations` to choose another file. |
| `artifacts/freeze/manifest.json` | Path interventions. Pins resource file checksums, graph and training hashes, retrieval settings, and target, judge and embedding identities. |

Record and request schemas are defined in [contracts.py](src/specalgo/contracts.py). Per-file checksums hash the original bytes. Record and graph hashes use canonical JSON and preserve list order. Both use SHA-256. The implementation is in [common.py](src/specalgo/common.py).

For graph construction, supply training-only L2 and L3 catalogues and record-to-rule mappings under `data/hierarchy/`. Call `specalgo.graph.build_graph(Dataset().load())` with `Dataset` imported from `specalgo.data`. The builder writes JSON and GraphML under the work directory. Copy the chosen baseline JSON into your resource directory when preparing a snapshot for inference.

The freeze manifest's `files` object maps resource-relative paths to SHA-256 checksums. It must cover at least the training records, split manifest and baseline graph. The other required fields are `graph_hash`, `train_hash`, `retrieval`, `target_model`, `judge_model` and `embedding`. Compute `freeze_id` with `specalgo.common.digest` over the manifest before adding that field. The runner validates the input fingerprint and records the installed protocol's fingerprint separately.

## Run a request

Save the following as `request.json` and choose a query relevant to your own graph.

```json
{"schema_version": "1.0", "operation": "retrieve", "query": "Which rules apply when a source is uncertain?", "top_k": 5}
```

```sh
specalgo alignment-value --resources ./resources --work-dir ./runs \
  --request request.json --output ./runs/retrieval.json
```

For an answer, change the operation to `align`. Add the matching `case_id` and `query_type` to obtain a score. For `represent`, supply a training `case_id` or omit it to process the training split. To limit a batch, pass `--selection` with a JSON object containing an ordered `selected_case_ids` array.

For a path intervention, replace the placeholder with an evaluation case ID, save the request as `paths.json` and run the second command.

```json
{
  "schema_version": "1.0",
  "case_ids": ["YOUR_PAIRED_CASE_ID"],
  "operations": ["remove_content", "remove_edge", "rewire_edge", "restore"]
}
```

```sh
specalgo path-influence --resources ./resources --work-dir ./runs \
  --request paths.json --output ./runs/influence.json
```

`--output` receives the result. The work directory holds derived indexes, caches and run evidence, including local prompts and generated answers. Keep that directory with your private inputs. The source distribution contains no corpus, graph snapshot, model weights or experiment results.

## Licensing

The SpecBench evaluation prompt retains its upstream Apache License 2.0 terms. [Upstream notice](src/specalgo/_support/scoring/SpecBench_LICENSE)

A project-wide license is pending selection by the rights holders.
