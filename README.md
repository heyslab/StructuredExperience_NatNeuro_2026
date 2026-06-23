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

Some manuscript analyses also used packages such as scikit-learn, JAX, and
spikeinterface, but those analyses are not all included in this minimal RNN
training release.

## External Code and Data Expectations

The scripts expect the lab's model database helper module:

```python
import models_database as mdb
```

That module is maintained separately from this repository. In the original
environment, it provides model/task/project lookup, database insertion, model
attribute retrieval, and saved-model path tracking.

The trained networks and animal behavior/electrophysiology data are also
handled outside this small source-code release. The manuscript's code and data
availability statements describe those resources separately.

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

1. `models_database` is a separate local dependency. The training scripts expect
   it to be importable in the Python environment used to run these files.

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

## AI Assistance Disclosure

This README draft was generated with AI assistance using the manuscript text and
the provided source files as context. The source code included in this
repository was not generated by AI.
