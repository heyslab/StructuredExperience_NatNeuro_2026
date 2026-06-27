# Structured Experience RNN Code

This repository contains the custom recurrent neural network (RNN) training and
analysis code accompanying the manuscript:

**Structured experience shapes strategy learning and neural dynamics in the
medial entorhinal cortex**

The code is provided as an archival record of the modeling and analysis
workflow used for the manuscript. It includes the core RNN training scripts,
the custom task/model classes, helper plotting code, curated figure-generation
scripts for the main and extended-data figures, and small cached summary files
where those were practical to include directly.

## Manuscript Context

The manuscript studies how prior structured experience changes later learning
and neural dynamics in a temporal delayed non-match-to-sample task (tDNMS).
Trials contain two timed odor cues separated by an inter-stimulus interval.
Short-Long (SL) and Long-Short (LS) trials are Go trials, while Short-Short
(SS) trials are No-Go trials.

The RNNs model this task with:

- A trial-start input, corresponding to a start light cue.
- A timed cue input, corresponding to odor presentation.
- A fully connected recurrent layer of 128 leaky units.
- A single output node representing the response signal.
- Noisy inputs and private unit noise during training.
- Mean-squared-error training against the target response window.

The central comparison is between networks trained directly on the full task
and networks first exposed to a shaping curriculum. In the standard shaping
curriculum, networks are first trained on the non-match structure before SS
No-Go trials are introduced.

Common shorthand used below:

```text
NS      No shaping before full-task training
S/FT    Standard SL+LS non-match shaping followed by full-task training
SL/FT   SL-only shaping followed by full-task training
LS/FT   LS-only shaping followed by full-task training
MEC     Medial entorhinal cortex
```

## Repository Layout

```text
run_rnn_base.py                      Core RNN training script
seed_weights_run.py                  Eigenspectrum-seeded training script
classes/                             Task generation, model layers, callbacks
analysis_tools/                      Plotting/progress helper code used here
ms_figures/                          Manuscript figure analysis scripts
fpf_code/                            Fixed-point precomputation utility
manuscript_tables.sql                SQL dump of manuscript model metadata
```

The `ms_figures/` directory preserves the original figure-oriented organization
of the analysis code. Most scripts are intended to be run as standalone
analysis scripts in an environment with access to the trained model archive or
processed electrophysiology/session files.

## External Resources and Data Layout

Many scripts import the companion model metadata helper:

```python
import models_database as mdb
```

For this public release, `models_database` is available at:

https://github.com/heyslab/models_database-public

That repository contains the public helper code expected by the scripts. This
repository also includes `manuscript_tables.sql`, a sanitized SQL dump of the
model metadata tables used for the manuscript. These resources document the
model IDs, task labels, project labels, saved-model paths, and parameter
attributes referenced by the analysis scripts.

The code reflects the original lab analysis environment. In particular:

- Many model-analysis scripts obtain trained-model paths from
  `models_database`.
- Several model-analysis scripts expect files stored beside trained model
  directories, such as `fps_info.pkl`, `<model_path>.dat.h5`, or
  `model.<model_id>_time_decoding.h5`.
- Mouse behavior and electrophysiology scripts expect processed per-session
  files such as `behavior.h5`, `spikes.h5`, `analysis/time_decoding.h5`, and
  `analysis/tangling.h5`.
- Figure outputs are commonly written to absolute paths under
  `/analysis/ms_figures/`.

Users running the scripts outside the original analysis environment may need to
adjust path lists, mount equivalent data directories, or provide local copies
of the expected trained-model and processed-session files.

Small cached files included directly in this repository are:

```text
ms_figures/testing_data.hdf
ms_figures/noisy_errors_allmodels.hdf
ms_figures/jitter_errors.hdf
ms_figures/extra_short_data.hdf
ms_figures/shaping_tasks_testing_data.hdf
ms_figures/seed_training/testing_data.hdf
ms_figures/time_decoding/time_decoding.h5
ms_figures/probe_mice/shuffle_results.csv
ms_figures/ephys_behvior/*.csv
```

Large trained models, per-session electrophysiology files, and most
per-model/per-session analysis caches are not included directly in the GitHub
repository.

The associated data and model archive is available on Zenodo:

https://doi.org/10.5281/zenodo.20855878

## Software Requirements

The original scripts were written for the lab Python environment and use
standard scientific Python packages plus local helper modules:

```text
python
tensorflow / keras
jax
numpy
pandas
PyTables or another pandas HDF5 backend
matplotlib
scipy
scikit-learn
statsmodels
pingouin
PyYAML
pyzmq
```

## Conda Environment Files

The original analyses used separate Conda environments for TensorFlow/Keras
model training and JAX-based fixed-point analysis. Environment export files are
included in:

```text
conda_envs/environment-tensorflow.yml
conda_envs/environment-tensorflow-full.yml
conda_envs/environment-jax.yml
conda_envs/environment-jax-full.yml
```

The shorter files were exported with `conda env export --from-history` and are
the recommended starting point for recreating portable environments:

```bash
conda env create -f conda_envs/environment-tensorflow.yml
conda env create -f conda_envs/environment-jax.yml
```

The `*-full.yml` files are included as archival records of the exact lab
environments used during analysis. They may be less portable across operating
systems or CUDA/library versions, but can be useful when matching the original
software stack closely.

Use the TensorFlow environment for model training and most plotting scripts
that load trained Keras models. Use the JAX environment for fixed-point finder
workflows and scripts under `fpf_code/` or `ms_figures/fp_analysis/` that
perform JAX-based fixed-point calculations.

The custom RNN layer uses TensorFlow/Keras internals from the environment in
which the models were developed. Reusing the code in a new environment may
require matching package versions.

## Core Training Code

### `run_rnn_base.py`

Primary training script for the base RNN models. It can initialize a new leaky
RNN or load an existing model by database ID, generate tDNMS-style training
data, train the network, save model outputs, and update the model database.

Example use in the original environment:

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

Training script for the eigenspectrum intervention described in the
manuscript. It loads recurrent-weight eigenspectra from previously trained
S/FT models, uses the leading shaping-associated eigenvalues as a scaffold,
and applies those values during early training of otherwise unshaped networks.

Example use in the original environment:

```bash
python seed_weights_run.py --task just_short_match --epochs 1000 --units 128
```

### `classes/`

The `classes/` package contains the task-generation and model code:

```text
classes/models.py      LeakyRNNCell, LeakyRNN, and Baseline model classes
classes/datagen.py     tDNMS, shaping, altered-shaping, and probe task generators
classes/callbacks.py   Keras callbacks used during training
classes/utils.py       ModelSaver and related save utilities
```

Task names used by the scripts must correspond to branches in
`genFactory.create(...)` in `classes/datagen.py`.

## Figure Analysis Scripts

The following sections list the curated scripts included for each manuscript
figure. Schematic panels, histology image panels, and some assembled layout
elements from the manuscript are not generated by Python scripts in this
release.

### Figure 1

Figure 1 introduces the RNN task model and compares NS and S/FT networks on
training, example unit activity, task performance, input-noise robustness, and
cue-timing jitter robustness.

```text
ms_figures/avg_training.py              # Figure 1c
ms_figures/example_units.py             # Figure 1d,f
ms_figures/example_units_no_shaping.py  # Figure 1e,g
ms_figures/model_performance.py         # Figure 1h,i
ms_figures/noisy_errors_allmodels.py    # Figure 1j,k
ms_figures/jitter_errors_allmodels.py   # Figure 1l,m
```

Included local caches:

```text
ms_figures/testing_data.hdf
ms_figures/noisy_errors_allmodels.hdf
ms_figures/jitter_errors.hdf
```

### Figure 2

Figure 2 analyzes how shaping changes recurrent state-space structure. The
scripts generate PCA projections, explained-variance summaries, trajectory
tangling examples, and within- vs across-context time-decoding plots.

```text
ms_figures/pca_states.py                         # Figure 2a
ms_figures/explained_var.py                      # Figure 2b
ms_figures/tangling_example.py                   # Figure 2c
ms_figures/time_decoding/time_decoding_examples.py  # Figure 2d,f
ms_figures/time_decoding/time_decoding_plots.py     # Figure 2e,g
```

Included local cache:

```text
ms_figures/time_decoding/time_decoding.h5
```

### Figure 3

Figure 3 analyzes recurrent connectivity. These scripts summarize learned
weight structure, eigenspectrum differences between training conditions,
decoding of shaping history from eigenvalue features, and the seeded-weight
intervention that biases new networks toward S/FT-like behavior.

```text
ms_figures/avg_weights.py                            # Figure 3a,b,c
ms_figures/weights_analysis/weights_eigs_figure.py   # Figure 3d,e
ms_figures/weights_analysis/weights_eigs_decoding.py # Figure 3f
ms_figures/seed_training/avg_training.py             # Figure 3g
ms_figures/seed_training/seed_run_pca.py             # Figure 3h
ms_figures/seed_training/seed_performance.py         # Figure 3i
```

Included local cache:

```text
ms_figures/seed_training/testing_data.hdf
```

### Figure 4

Figure 4 analyzes fixed/slow points and dynamical motifs. The scripts compare
flow fields, dynamic motifs, limit-cycle structure, and how changing cue timing
moves S/FT networks into response-potent regions of state space.

```text
ms_figures/fp_analysis/flowfield_figure.py          # Figure 4a,b,c
ms_figures/fp_analysis/dynamic_motifs.py            # Figure 4d
ms_figures/fp_analysis/limit_flows_pca_ns_figure.py # Figure 4e
ms_figures/fp_analysis/limit_flows_pca_figure.py    # Figure 4f
ms_figures/fp_analysis/morph_task.py                # Figure 4g,i
ms_figures/fp_analysis/morph_task_output_figure.py  # Figure 4h,j
```

The fixed-point/RNN helper code used by these analyses is included at:

```text
ms_figures/fp_analysis/leaky_rnn.py
ms_figures/fp_analysis/fixed_point_finder/fixed_points.py
```

The utility used to find and save fixed points is:

```text
fpf_code/find_and_save_fps.py
```

That utility includes a small vendored subset of the Apache-2.0
`computation-thru-dynamics` fixed-point finder code:

```text
fpf_code/computation-thru-dynamics/
```

### Figure 5

Figure 5 asks whether the choice of shaping task matters. It compares SL-only
and LS-only shaping with the standard S/FT procedure, including training
performance, example task responses, state-space structure, and final MSE.

```text
ms_figures/avg_training.py              # Figure 5b,f
ms_figures/shaping_tasks_performance.py # Figure 5c,g
ms_figures/shaping_pca.py               # Figure 5d
ms_figures/mse_comparison.py            # Figure 5h
```

Included local cache:

```text
ms_figures/shaping_tasks_testing_data.hdf
```

### Figure 6

Figure 6 compares RNN predictions with mouse behavior and MEC recordings. The
included scripts generate behavior summaries, example time-locked neural
activity, ephys time-decoding matrices, single-trial decoding summaries, and
pseudo-population PCA summaries.

```text
ms_figures/ephys_behvior/performance_plot.py                 # Figure 6b,d
ms_figures/ephys_time_decoding/spikes_by_cell_figure.py      # Figure 6e
ms_figures/ephys_time_decoding/test_ephys_decode.py          # Figure 6f,g,h
ms_figures/ephys_time_decoding/time_coding_single_trials.py  # Figure 6i
ms_figures/ephys_pca_plots/paper_figure.py                   # Figure 6j,k
```

Included local behavior-summary files:

```text
ms_figures/ephys_behvior/sl_only_avg.csv
ms_figures/ephys_behvior/sl_by_type.csv
ms_figures/ephys_behvior/tdnms_control.csv
ms_figures/ephys_behvior/ctrl_by_type.csv
```

The helper script that generates per-session `analysis/time_decoding.h5` files
from ephys `spikes.h5` files is:

```text
ms_figures/ephys_time_decoding/time_decoding_ephys.py
```

### Figure 7

Figure 7 tests RNN predictions on novel temporal probe trials in mice. Probe
trials are unrewarded LL, MM, or extended-cue configurations interleaved into
otherwise familiar tDNMS sessions. The scripts compare animal probe responses
with the corresponding RNN predictions and plot licking rasters for each probe
condition.

```text
ms_figures/probe_mice/probe_performance_andShuffle.py  # Figure 7b
ms_figures/probe_mice/probe_rnn_performance.py         # Figure 7c
ms_figures/probe_mice/allprobes_lick_raster.py         # Figure 7d,e,f,g
```

Included local summary file:

```text
ms_figures/probe_mice/shuffle_results.csv
```

## Extended Data Analysis Scripts

Note: Extended Data figures were renumbered during final manuscript
preparation to match their order of citation in the manuscript. The labels
below follow the final manuscript numbering.

### Extended Data Figure 1

Extended Data Figure 1 expands the task-performance and robustness analyses.
It quantifies response/error-window separation across RNNs and shows example
responses under increasing input noise and cue-timing jitter.

```text
ms_figures/quantify_errors_figure.py # Extended Data Figure 1a
ms_figures/noisy_error_compare.py    # Extended Data Figure 1b,c,d,e
```

Included local cache:

```text
ms_figures/extra_short_data.hdf
```

### Extended Data Figure 2

Extended Data Figure 2 expands the state-space and time-decoding analyses. It
shows PCA trajectories for multiple NS and S/FT networks, compares trajectory
tangling, and reuses the time-decoding example workflow for additional
SL-trained decoder examples.

```text
ms_figures/multimodel_pca.py                       # Extended Data Figure 2a,b
ms_figures/tangling.py                             # Extended Data Figure 2c
ms_figures/time_decoding/time_decoding_examples.py # Extended Data Figure 2d,e
```

### Extended Data Figure 3

Extended Data Figure 3 is generated by the dynamic-motifs script also used in
Figure 4. It compares fixed/slow-point structure and task trajectories across
NS, shaping-only, and S/FT models.

```text
ms_figures/fp_analysis/dynamic_motifs.py # Extended Data Figure 3a,b,c
```

### Extended Data Figure 4

Extended Data Figure 4 analyzes LS error trials. The script compares early
response magnitude, state-space trajectories, and the relationship between
distance to fixed points and error-like output in NS and S/FT networks.

```text
ms_figures/fp_analysis/error_trials_figure.py # Extended Data Figure 4a,b,c,d,e,f
```

### Extended Data Figure 5

Extended Data Figure 5 further analyzes the timer/detector interpretation of
S/FT dynamics. It highlights Cue On and Cue Off trajectories, response gating
regions, and responses to isolated ambiguous cue input.

```text
ms_figures/fp_analysis/limit_flows_pca_figure.py # Extended Data Figure 5a,b
ms_figures/fp_analysis/response_gating.py        # Extended Data Figure 5c
ms_figures/s_cue_test.py                         # Extended Data Figure 5d
```

### Extended Data Figure 6

Extended Data Figure 6 quantifies geometric features of the RNN dynamics,
including directional bias in the Cue Off mode and offsets between Cue On and
Cue Off fixed/slow-point structure.

```text
ms_figures/fp_analysis/geom_detecting_figure.py # Extended Data Figure 6a,b,c
```

### Extended Data Figure 7

Extended Data Figure 7 extends the dynamical-motif analysis to novel cue
configurations. One script tests omission of the first cue, and the other tests
untrained LL, MM, and extended-cue probe trials predicted to evoke responses in
S/FT networks.

```text
ms_figures/fp_analysis/dynamic_motifs_omit_one.py # Extended Data Figure 7a
ms_figures/fp_analysis/dynamic_motifs_probes.py   # Extended Data Figure 7b
```

### Extended Data Figure 8

Extended Data Figure 8 compares SL/FT and S/FT networks. The scripts show
within- and across-context time decoding for LS- and SL-trained decoders and
compare trajectory tangling between the two training histories.

```text
ms_figures/time_decoding/sl_time_decoding_ex.py # Extended Data Figure 8a,b,c
ms_figures/sl_analysis/tangling_sl.py           # Extended Data Figure 8d
```

The time-decoding script expects per-model files named:

```text
model.<model_id>_time_decoding.h5
```

### Extended Data Figure 9

Extended Data Figure 9 combines representative histology image panels with
behavioral and MEC analyses. The Python scripts included here generate the
all-mouse licking rasters, ephys tangling comparison, and SL-trained decoder
confusion matrices. The histology panels are not generated by the Python code.

```text
ms_figures/ephys_behvior/allbehavior_lickraster.py  # Extended Data Figure 9c,d
ms_figures/ephys_time_decoding/tangling_figure.py   # Extended Data Figure 9e
ms_figures/ephys_time_decoding/test_ephys_decode.py # Extended Data Figure 9f,g
```

The helper script that generates per-session `analysis/tangling.h5` files from
ephys `spikes.h5` files is:

```text
ms_figures/ephys_time_decoding/tangling_ephys.py
```

## Notes for Reuse

This release is intended to document the manuscript workflow and support
transparent inspection of the analyses. It is not packaged as a turnkey
software library. A few implementation details reflect the original analysis
environment:

1. `models_database` is external to this repository and is available at
   https://github.com/heyslab/models_database-public.
2. Model outputs default to `/models`, matching the original server setup.
3. The `--no_update` option in the training script is intended to suppress
   database updates, but some database path bookkeeping remains in the current
   scripts.
4. `seed_weights_run.py` uses a fixed set of manuscript model IDs to compute
   the shaping eigenspectrum scaffold.
5. `genFactory.create(...)` uses the manuscript training batch size internally,
   even though surrounding parameter dictionaries also carry a `batch_size`
   field.
6. Stored recurrent-weight logging assumes the manuscript model size of
   128 recurrent units.
7. Some analysis scripts contain absolute output paths and original session
   path lists. These may need to be adjusted outside the lab environment.

## AI Assistance Disclosure

This README was generated with AI assistance using the manuscript text and the
provided source files as context. The source code included in this repository
was not generated by AI.
