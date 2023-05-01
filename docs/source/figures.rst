Figures
=======

CEBRA was introduced in `Schneider, Lee and Mathis (2022)`_ and applied to various datasets across
animals and recording modalities.

In this section, we provide reference code for reproducing the figures and experiments. Since especially
experiments containing parameter sweeps take prohibitely long on a single machine, experiments are grouped
in two categories:

   * A set of demo notebooks for outlining the most common use cases of CEBRA. These notebooks can be run
     on a local or hosted `jupyter notebook` server, or be run in Google colab. To get started with using CEBRA
     on a set of worked examples, this is the place to start.
   * The collection of plotting code for all paper figures. The figures are generated from cached experimental
     results. For data (re-) analysis and performance comparisons of CEBRA, this is the easiest way to get started.
   * The collection of experiments for obtaining results for the figures. Experiments should ideally be run on 
     a GPU cluster with SLURM pre-installed for the best user experience. Alternatively, experiments can also be 
     manually scheduled (our submission system produces a stack of bash files which can be executed on any machine).
     We recommend this route for follow-up research, when CEBRA (or any of our baselines) should be used for
     comparisons against other methods.



List of paper figures 
---------------------

We provide reference code for plotting all paper figures here.
Note that for the paper version, panels might have been post edited, and the figures might 
differ in minor typographic details.

.. toctree::
   :caption: List of figures

   Figure 1 <cebra-figures/figures/Figure1.ipynb>
   Figure 2 <cebra-figures/figures/Figure2.ipynb>
   Figure 3 <cebra-figures/figures/Figure3.ipynb>
   Figure 4 <cebra-figures/figures/Figure4.ipynb>
   Figure 5 <cebra-figures/figures/Figure5.ipynb>

   Extended Data Fig. 4 <cebra-figures/figures/ExtendedDataFigure4.ipynb>
   Supplementary Table S1 and S2 <cebra-figures/figures/SupplTableS1S2.ipynb>
   Supplementary Table S3 and S4 <cebra-figures/figures/SupplTableS3S4.ipynb>
   Supplementary Table 5 <cebra-figures/figures/SupplTableS5.ipynb>
   Supplementary Table 6 <cebra-figures/figures/SupplTableS6.ipynb>



