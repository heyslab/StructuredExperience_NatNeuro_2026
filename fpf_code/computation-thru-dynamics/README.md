# Computation Through Dynamics Subset

This folder contains a small subset of the `computation-thru-dynamics` fixed
point finder code used by the manuscript analysis scripts.

The included files are redistributed under the Apache License, Version 2.0.
Copyright and license headers have been retained in the source files, and the
Apache 2.0 license text is included in this folder.

This is not a full copy of the upstream project. It includes only the fixed
point finder components needed to support the RNN fixed-point analyses in this
repository.

Small local housekeeping edits may have been made to the copied files, such as
removing interactive debugger breakpoints, stale local TODO notes, or comments
that were specific to development on the original analysis server. These edits
are intended only to make the released source easier to read and run; they are
not intended to change the fixed-point optimization algorithms.

For the manuscript analyses, this code is used by:

```text
fpf_code/find_and_save_fps.py
```

which computes and saves fixed points for trained RNN models referenced through
the companion model metadata.
