![Spectre](https://user-images.githubusercontent.com/1897842/31115297-0fe2c3aa-a822-11e7-90e6-92ceccf76137.jpg)

# pca-exims-worker
Implementation of PCA transformation supported by EXIMS score for components selection as a worker of Spectre system.


# EXIMS

Is a measure of a grayscale image structness. Helps to distinguish, whether
image is just noise, or carries some information. Implementation based on:
[Wijetunge, Chalini & Saeed, Isaam & Boughton, Berin & Spraggins, Jeffrey & M Caprioli, Richard & Bacic, Antony & Roessner, Ute & Halgamuge, Saman. (2015). EXIMS: An improved data analysis pipeline based on a new peak picking method for EXploring Imaging Mass Spectrometry data. Bioinformatics (Oxford, England). 31. 10.1093/bioinformatics/btv356.](https://www.researchgate.net/publication/278042229_EXIMS_An_improved_data_analysis_pipeline_based_on_a_new_peak_picking_method_for_EXploring_Imaging_Mass_Spectrometry_data)


# Knee Estimation

For estimation of number of EXIMS-confirmed components, Kneedle algorithm is
used, with [kneed](https://github.com/arvkevi/kneed) implementation. Original
paper: [Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior Ville Satopa † , Jeannie Albrecht† , David Irwin‡ , and Barath Raghavan§ †Williams College, Williamstown, MA ‡University of Massachusetts Amherst, Amherst, MA § International Computer Science Institute, Berkeley, CA](https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf)
