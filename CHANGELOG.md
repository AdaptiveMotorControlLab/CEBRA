# Changelog

When contributing a PR, please add the title, link and a short 1-2 line description of the 
PR to this document. If you are an external contributor, please also add your github handle.
You can use markdown formatting in this document.

**Template for contribution summaries**: Please use the following to extend the changelog:

```
- **The PR title [#<number>](https://github.com/<user>/<repo>/pull/<number>)**:
  <Short 1-2 line description of the PR>
```

**Info for maintainers**: When creating a new release, make sure to update the `latest` heading
in this file to the released code version using the name of the github tag (e.g. `v0.1.2`,
`v0.1.2a3`, `v0.1.2b3`, etc.).

- **Add embedding ensembling functionality [#507](https://github.com/AdaptiveMotorControlLab/CEBRA-dev/pull/507)**:
  Add ``ensemble_embeddings`` that aligns multiple embeddings and combine them into an averaged one.

- **Move `max_validation_iterations` from `cebra.CEBRA` to `cebra.metrics.infonce_loss` [#527](https://github.com/AdaptiveMotorControlLab/CEBRA-dev/pull/527)**:
  Move `max_validation_iterations` from `cebra.CEBRA` to `cebra.metrics.infonce_loss` and 
  rename the variable to `num_batches`. 

- **Add `plot_consistency` and demo notebook [#502](https://github.com/AdaptiveMotorControlLab/CEBRA-dev/pull/502)**:
  Add `plot_consistency` helper function and complete the corresponding notebook.


## v0.0.3rc1

- **Add helpers to use DeepLabCut data with CEBRA [#436](https://github.com/stes/neural_cl/pull/436)**:
  Add helpers to preprocess DeepLabCut output data and use it easily with CEBRA.
- **Add `compare_models` functionality [#460](https://github.com/stes/neural_cl/pull/460)**:
  Multiple trained models can now be plotted together for easier comparison of hyperparameter
  settings and datasets.

## v0.0.2 (10/02/23)

This release contains various additions from the work on three successive release candidates.
It is the official first release distributed along with the publication of the CEBRA paper.

- v0.0.2rc3
  - **Add adapt=True in CEBRA.fit() [#445](https://github.com/stes/neural_cl/pull/445)**:
    Add capability to adapt a trained CEBRA models to new sessions of data, potentially with different input
    dimensions.
  - **Save/load functionality for sklearn models [#408](https://github.com/stes/neural_cl/pull/408)**:
    Add a `save/load` function to `cebra.CEBRA` for serialization. Experimental feature for now which will be
    refined later on.
- v0.0.2rc2
  - **Add cebra.plot package [#385](https://github.com/stes/neural_cl/pull/385)**:
    Simplify post-hoc analysis of model performance and embeddings by collecting plotting functions for the most common usecases.
  - **Multisession API integration [#333](https://github.com/stes/neural_cl/pull/333)**:
    Add multisession implementation compatibility to the sklearn API. 
- v0.0.2rc1
  - **Implementation for general dataloading [#305](https://github.com/stes/neural_cl/pull/305)**:
    Implement `load`, a general function to convert any supported data file types to ``numpy.array``.
  - **Add score method [#316](https://github.com/stes/neural_cl/pull/316)**:
    Add ``score`` method to ``cebra`` to compute the score of the trained model on new data.
  - **Add quick testing option [#318](https://github.com/stes/neural_cl/pull/318)**:
    Add slow marker for longer tests and a quick testing option for pytest and in github workflow.
  - **Add CITATION.cff file [#339](https://github.com/stes/neural_cl/pull/339)**:
    Add CITATION.cff file for easy-to-use citation of the pre-print paper. 
  - **Update sklearn dependency [#317](https://github.com/stes/neural_cl/pull/317)**:
    The sklearn dependency was updated to `scikit-learn` as discussed
    [in the scikit-learn docs](https://github.com/scikit-learn/sklearn-pypi-package)
  - **Increase documentation coverage >90% [#265](https://github.com/stes/neural_cl/pull/265)**:
    Configure `interrogate` for checking docstring coverage of the codebase. Add docstrings to increase
    overall coverage to >90%.
  - **Increase documentation coverage >80% [#263](https://github.com/stes/neural_cl/pull/263)**:
    Configure `interrogate` for checking docstring coverage of the codebase. Add docstrings to increase
    overall coverage to >80%.
  - **Apply new code and docstring formatting to whole codebase [#255](https://github.com/stes/neural_cl/pull/255)**:
    Before enforcing google style doc strings with `yapf`, apply `black` for stricter code formatting.
    Format docstrings with `docformatter`.
  - **Run formatter during workflow run [#217](https://github.com/stes/neural_cl/pull/217)**:
    This addition checks that `make docs` can be run as part of the tests.
  - **Update documentation and enforce working links [#198](https://github.com/stes/neural_cl/pull/198)**:
    Revision and improvement of the current documentation. "nitpicky" mode is now used in sphinx,
    which will check that we dont have any broken links of missing references in the documentation.

## v0.0.1 (21/09/22)

- Version of the code submitted along with the paper revision
