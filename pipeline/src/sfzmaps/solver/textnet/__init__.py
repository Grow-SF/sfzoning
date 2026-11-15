"""The text detector: a small U-Net that scores every px of a 2000 sheet's page as printed text or not, at the
scan's own resolution, trained only on Terra's saved text boxes.  One model per quarter of the city, each trained
with its quarter held out (`sfzmaps solver text Q`, stage `solver-text-<Q>`), and their scores on the quarters they
never saw (`sfzmaps solver text-scores`, stage `solver-text-scores`).

Modules:
  data      each sheet's page, what Terra marked as text there, what is unknown, and each px's quarter
  model     the U-Net
  train     fitting one quarter's model
  predict   its text chance over every sheet's page
  stage     one quarter's stage: training and scoring in one turn on the GPU
  evaluate  each model's scores on the quarter it never saw, from its chances
"""
