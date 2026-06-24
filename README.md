# Structured Experience RNN Code

This repository contains the recurrent neural network (RNN) training code used
for the manuscript:

**Structured experience shapes strategy learning and neural dynamics in the
medial entorhinal cortex**

The code is provided as a record of the custom modeling workflow
used in the study.

## Manuscript Context

The manuscript studies how structured prior experience changes later learning
and neural dynamics in a temporal delayed non-match-to-sample task (tDNMS).
In the task, trials contain two timed odor cues separated by an inter-stimulus
interval. Short-Long (SL) and Long-Short (LS) trials are Go trials, while
Short-Short (SS) trials are No-Go trials.

The RNNs in this repository model that task with:

- A start cue input, corresponding to the trial-start light cue.
- A timed cue input, corresponding to odor presentation.
- A fully connected recurrent layer of leaky units.
- A single output node representing the response signal.
- Noisy inputs and private unit noise during training.
- Mean-squared-error training against the target response window.

The main comparison in the manuscript is between RNNs trained directly on the
full task and RNNs first exposed to a shaping curriculum. The shaping phase
omits SS No-Go trials and trains networks on the non-match trial structure
before introducing the full tDNMS task. The manuscript uses these models to
study performance, state-space structure, recurrent connectivity, dynamical
motifs, and predictions tested in mouse behavior and MEC recordings.

## Repository Contents

```text
run_rnn_base.py
seed_weights_run.py
classes/
  callbacks.py
  datagen.py
  models.py
  utils.py
```

### `run_rnn_base.py`

Primary training script for the base RNN models. It can initialize a new leaky
RNN or load an existing model by database ID, generate tDNMS-style training
data, train the network, save model outputs, and update the model database.

Typical use in the original environment:

```bash
python run_rnn_base.py --task just_short_match --epochs 1000 --units 128
```

Useful options include:

```text
--task / -t          Task name defined in classes/datagen.py
--base_id / -b       Existing model ID to initialize from
--epochs / -e        Number of training epochs
--units / -u         Number of recurrent units
--create_task        Add a new task entry to the model database
--create_project     Add a new project entry to the model database
--no_train           Save an initialized/untrained model
--store_weights      Store recurrent weights during training
--no_update / -X     Intended for runs that should not update the database
```

### `seed_weights_run.py`

Training script for the eigenspectrum intervention described in the manuscript.
It loads recurrent-weight eigenspectra from previously trained shaped models,
uses the leading shaping-associated eigenvalues as a scaffold, and applies
those values during early training of otherwise unshaped networks.

Typical use in the original environment:

```bash
python seed_weights_run.py --task just_short_match --epochs 1000 --units 128
```

This script is most relevant to the manuscript analysis showing that recurrent
connectivity features associated with shaping can bias later network behavior.

### `classes/models.py`

Defines the custom leaky recurrent layer used by the training scripts:

- `LeakyRNNCell`
- `LeakyRNN`
- `Baseline`

The model modifies a TensorFlow/Keras SimpleRNN-style cell by adding a leak
parameter and private unit noise.

### `classes/datagen.py`

Defines trial generators for the tDNMS task and related shaping/probe variants,
including:

- `tDNMSGenerator`
- `tDNMSGeneratorLL`
- `SLOnlyTask`
- `LSOnlyTask`
- `tDNMS_no_match`
- `CueResponse`
- `tDNMSRespAll`
- `genFactory`

The `genFactory.create(...)` method maps task-name strings to the corresponding
trial generator.

### `classes/callbacks.py`

Defines Keras callbacks used during training:

- `LoggerCallback` records predictions and optionally recurrent weights.
- `UntrainedCB` saves an initialized model and stops training immediately.

### `classes/utils.py`

Defines `ModelSaver`, which writes trained models, validation data, training
history, predictions, parameters, and optional recurrent weights.

## Software Requirements

The original scripts were written for the lab training environment. They rely
on standard scientific Python tools plus local helper code:

```text
python
tensorflow / keras
numpy
pandas
PyYAML
PyTables or another pandas HDF5 backend
pyzmq
```

```markdown
The included figure scripts additionally use plotting/statistics packages such
as `matplotlib`, `scipy`, `statsmodels`, `scikit-learn`, and `pingouin`.

## External Model Database Metadata

The training scripts import:

```python
import models_database as mdb 
```

For this release, the companion metadata helper is available here:

https://github.com/heyslab/models_database-public/tree/main

That repository contains the public `models_database` helper code and a
sanitized SQL dump of the model metadata tables used for the manuscript. The 
dump is provided to document the model IDs, task labels, project labels,
saved-model paths, and parameter attributes referenced by the training scripts.

This companion repository is intended as an archival transparency resource for 
the manuscript, not as a supported database package.

## Example Workflow

In the original lab environment, a typical shaped-vs-unshaped comparison was:

1. Train or initialize a model for a task defined in `classes/datagen.py`.
2. Save the model and metadata through `ModelSaver` and `models_database`.
3. Analyze model performance, recurrent activity, eigenspectra, and state-space
   dynamics using downstream analysis scripts.
4. Compare RNN predictions with mouse behavior and MEC recordings described in
   the manuscript.

For example:

```bash
python run_rnn_base.py --task just_short_match --epochs 1000 --units 128
python run_rnn_base.py --task no_match --epochs 300 --units 128
python seed_weights_run.py --task just_short_match --epochs 1000 --units 128
```

Task names must correspond to branches in `genFactory.create(...)`.

## Notes for Readers

This repository is an archival code release accompanying the manuscript. A few
implementation details reflect the original analysis environment:

1. `models_database` is provided separately at
   https://github.com/heyslab/models_database-public/tree/main. It documents the 
   manuscript model metadata and preserves the API expected by these scripts.
   A SQL dump reflecting the database state at the time of publication is included in this
   repository.

2. Model outputs default to `/models`, matching the original server setup. Users
   running elsewhere may want to adjust this path or provide an equivalent
   mount point.

3. The `--no_update` option is intended to suppress database updates, but a
   small amount of database path bookkeeping is still present in the current
   scripts.

4. `seed_weights_run.py` uses a fixed set of model IDs to compute the shaping
   eigenspectrum scaffold. Those IDs refer to the original model database.

5. The `--no_train` path in `seed_weights_run.py` was not the primary use case
   for the manuscript eigenspectrum intervention workflow.

6. `genFactory.create(...)` currently uses the manuscript training batch size
   of 2 internally, even though the surrounding parameter dictionaries also
   carry a `batch_size` field.

7. Stored recurrent-weight logging assumes the manuscript model size of
   128 recurrent units. This matches the reported experiments.

8. The custom RNN layer uses Keras/TensorFlow internals from the environment in
   which the models were developed. Reusing the code in a new environment may
   require matching package versions.

## Scope of This Release

This repository is focused on the custom RNN training code. It does not yet
include every downstream plotting, fixed-point, electrophysiology-processing, or
statistical analysis script used to generate the full manuscript. The main
figures and extended data include additional analyses of:

- RNN task training and performance.
- Abstract time representations after shaping.
- Recurrent connectivity and eigenspectrum structure.
- Dynamical motifs and fixed/slow-point structure.
- Alternative shaping curricula.
- Mouse behavior and MEC recordings.
- Probe-trial predictions and behavioral validation.

## Included Figure Analysis Scripts

This release currently includes the core RNN training code and the manuscript
analysis scripts for Figure 1, 2 and 3. Additional figure scripts may be added as the 
release is curated figure-by-figure.

### Figure 1

Figure 1 scripts are in `ms_figures/`:

```text
ms_figures/avg_training.py              # Figure 1c
ms_figures/example_units.py             # Figure 1d,f
ms_figures/example_units_no_shaping.py  # Figure 1e,g
ms_figures/model_performance.py         # Figure 1h,i
ms_figures/noisy_errors_allmodels.py    # Figure 1j,k
ms_figures/jitter_errors_allmodels.py   # Figure 1l,m
```

The following small cached data files are included to support the Figure 1
analysis scripts:

```text
ms_figures/testing_data.hdf
ms_figures/noisy_errors_allmodels.hdf
ms_figures/jitter_errors.hdf
```

### Figure 2

Figure 2 scripts are in `ms_figures/`:

```text
ms_figures/pca_states.py                         # Figure 2a
ms_figures/explained_var.py                      # Figure 2b
ms_figures/tangling_example.py                   # Figure 2c
ms_figures/time_decoding/time_decoding_examples.py  # Figure 2d,f
ms_figures/time_decoding/time_decoding_plots.py     # Figure 2e,g
```

The following cached data file is included to support the Figure 2
time-decoding example panels:

```text
ms_figures/time_decoding/time_decoding.h5
```

### Figure 3

Figure 3 scripts are in `ms_figures/` and related subfolders:

```text
ms_figures/avg_weights.py                            # Figure 3a,b,c
ms_figures/weights_analysis/weights_eigs_figure.py   # Figure 3d,e
ms_figures/weights_analysis/weights_eigs_decoding.py # Figure 3f
ms_figures/seed_training/avg_training.py             # Figure 3g
ms_figures/seed_training/seed_run_pca.py             # Figure 3h
ms_figures/seed_training/seed_performance.py         # Figure 3i
```

These scripts analyze the recurrent connectivity and eigenspectrum structure of
the manuscript RNNs, including the shaped-model weight profiles, eigenvalue
distributions, decoding from eigenspectrum features, and the seed-weight
training intervention.

The fixed-point helper used
by the connectivity/eigenspectrum analyses is included here:

```text
ms_figures/fp_analysis/leaky_rnn.py
```

The following cached data file is included to support the Figure 3 seed-training
behavior panel:

```text
ms_figures/seed_training/testing_data.hdf
```

### Figure 4

Figure 4 scripts are in `ms_figures/fp_analysis/`:

```text
ms_figures/fp_analysis/flowfield_figure.py              # Figure 4a,b,c
ms_figures/fp_analysis/dynamic_motifs.py                # Figure 4d, S3a,b,c
ms_figures/fp_analysis/limit_flows_pca_ns_figure.py     # Figure 4e
ms_figures/fp_analysis/limit_flows_pca_figure.py        # Figure 4f, S6a,b
ms_figures/fp_analysis/morph_task.py                    # Figure 4g,i
ms_figures/fp_analysis/morph_task_output_figure.py      # Figure 4h,j
```

These scripts analyze fixed and slow points in the trained RNNs, including
state-space flow fields, dynamic motifs, limit-cycle structure, and the model
response to morphing the timing structure of the task.

The Figure 4 scripts use trained model paths and metadata from the companion
`models_database` release. Several of the scripts also expect fixed-point
results saved next to each trained model as: 

```text
fps_info.pkl
```

The fixed-point helper code used by these analyses is included here:

```text
ms_figures/fp_analysis/leaky_rnn.py
ms_figures/fp_analysis/fixed_point_finder/fixed_points.py
```

The utility used to find and save fixed points is included separately:

```text
fpf_code/find_and_save_fps.py
```

That utility includes a small vendored subset of the Apache-2.0
`computation-thru-dynamics` fixed-point finder code:

```text
fpf_code/computation-thru-dynamics/
```

Only the fixed-point finder components needed for the manuscript RNN analyses
are included. The Apache 2.0 license text and original source-file notices are 
retained with that code.

### Figure 5

Figure 5 scripts are in `ms_figures/`:

```text
ms_figures/avg_training.py                  # Figure 5b,f
ms_figures/shaping_tasks_performance.py     # Figure 5c,g
ms_figures/shaping_pca.py                   # Figure 5d
ms_figures/mse_comparison.py                # Figure 5h
```

These scripts analyze alternative shaping curricula and their effects on model
training, task performance, and recurrent state-space structure. The
`avg_training.py` script is shared with Figure 1 and also generates the Figure 5
training-summary panels.

The following cached data file is included to support the Figure 5 shaping-task
performance panels:

```text
ms_figures/shaping_tasks_testing_data.hdf
```

The Figure 5 scripts use trained model paths, task labels, and model attributes
from the companion `models_database` release. `avg_training.py` also reads
per-model training-history files from the trained-model output archive, named
like:

```text
<model_path>.dat.h5
```

As in the original analysis environment, these scripts write outputs to
absolute paths under `/analysis/ms_figures/`. Users running the scripts on a
different system may need to adjust those output paths or create equivalent
directories.

### Figure 6

Figure 6 scripts are in `ms_figures/ephys_behvior/`,
`ms_figures/ephys_time_decoding/`, and `ms_figures/ephys_pca_plots/`:

```text
ms_figures/ephys_behvior/performance_plot.py                 # Figure 6b,d
ms_figures/ephys_time_decoding/spikes_by_cell_figure.py      # Figure 6e
ms_figures/ephys_time_decoding/test_ephys_decode.py          # Figure 6f,g,h
ms_figures/ephys_time_decoding/time_coding_single_trials.py  # Figure 6i
ms_figures/ephys_pca_plots/paper_figure.py                   # Figure 6j,k
```

These scripts analyze mouse behavior and MEC electrophysiology data from the
structured-experience experiments. They generate the behavioral performance
summary, example neural responses, neural time-decoding analyses, and
population-state summaries shown in Figure 6.

The behavior summary script uses the following small CSV files, included next
to the script:

```text
ms_figures/ephys_behvior/sl_only_avg.csv
ms_figures/ephys_behvior/sl_by_type.csv
ms_figures/ephys_behvior/tdnms_control.csv
ms_figures/ephys_behvior/ctrl_by_type.csv
```

The ephys decoding and population-state scripts depend on processed
electrophysiology files generated outside this repository, including per-session
`spikes.h5`, `behavior.h5`, and `analysis/time_decoding.h5` files. These large
processed data files are not included directly in the GitHub repository.

The helper script used to generate the per-session ephys time-decoding files is
included here:

```text
ms_figures/ephys_time_decoding/time_decoding_ephys.py
```

It expects a session directory containing:

```text
spikes.h5
```

and writes:

```text
analysis/time_decoding.h5
```

within that session directory.

As in the original analysis environment, the Figure 6 scripts contain absolute
paths to the processed behavior/ephys data and write outputs to absolute paths
under `/analysis/ms_figures/`. Users running the scripts on a different system
may need to adjust those paths or provide equivalent directory structures.

### Figure 7

Figure 7 scripts are in `ms_figures/probe_mice/`:

```text
ms_figures/probe_mice/probe_performance_andShuffle.py  # Figure 7b
ms_figures/probe_mice/probe_rnn_performance.py         # Figure 7c
ms_figures/probe_mice/allprobes_lick_raster.py         # Figure 7d,e,f,g
```

These scripts analyze the mouse probe-trial behavior and compare the observed
probe responses with the corresponding RNN predictions.

The following small cached summary file is included to support the Figure 7
probe-behavior panel:

```text
ms_figures/probe_mice/shuffle_results.csv
```

The Figure 7 scripts also depend on processed behavior files generated outside
this repository. In the original analysis environment, these include:

```text
/data3/jack/goPercentages.csv
/heys-nas-LabData/Cambria/Jack Project - tDNMS to probe (MM, LL)/BaselineAndProbes/*/*/behavior.h5
```

The RNN probe-performance script uses trained model paths and metadata from the
companion `models_database` release.

### Extended Data Figure 1

Extended Data Figure 1 scripts are in `ms_figures/`:

```text
ms_figures/quantify_errors_figure.py  # Extended Data Figure 1
ms_figures/noisy_error_compare.py     # Extended Data Figure 1b,c,d,e
```

These scripts quantify error-like response peaks in trained RNNs and compare
model responses under increased input-noise conditions.

The following cached data file is included to support the noise-comparison
panels:

```text
ms_figures/extra_short_data.hdf
```

### Extended Data Figure 2

Extended Data Figure 2 scripts are in `ms_figures/` and
`ms_figures/time_decoding/`:

```text
ms_figures/multimodel_pca.py                         # Extended Data Figure 2a,b
ms_figures/tangling.py                               # Extended Data Figure 2c
ms_figures/time_decoding/time_decoding_examples.py   # Extended Data Figure 2d,e
```

These scripts show state-space examples across multiple trained RNNs, quantify
trajectory tangling across shaping conditions, and reuse the time-decoding
example workflow from Figure 2 for additional example panels.

The following cached data file is included to support the time-decoding example
panels:

```text
ms_figures/time_decoding/time_decoding.h5
```

### Extended Data Figure 3

Extended Data Figure 3 is generated by a script already listed with Figure 4:

```text
ms_figures/fp_analysis/dynamic_motifs.py  # Extended Data Figure 3a,b,c
```

This script analyzes fixed/slow-point structure and dynamic motifs in the
trained RNNs. It also contributes to Figure 4d.

The script uses trained model paths and metadata from the companion
`models_database` release, along with precomputed fixed-point files stored next
to each trained model as:

```text
fps_info.pkl
```

### Extended Data Figure 4

Extended Data Figure 4 is generated by:

```text
ms_figures/fp_analysis/geom_detecting_figure.py  # Extended Data Figure 4a,b,c
```

This script analyzes geometric features of the trained RNN state space and
relates directional flow structure to fixed/slow-point organization.

The script uses trained model paths and metadata from the companion
`models_database` release, along with precomputed fixed-point files stored next
to each trained model as:

```text
fps_info.pkl
```

It also uses the fixed-point/RNN helper code included with the Figure 4
analysis scripts:

```text
ms_figures/fp_analysis/leaky_rnn.py
```

## AI Assistance Disclosure

This README draft was generated with AI assistance using the manuscript text and
the provided source files as context. The source code included in this
repository was not generated by AI.
