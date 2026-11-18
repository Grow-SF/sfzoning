"""The text reader: a CTC recogniser of the 2000 sheets' printed words, read in our own word boxes (spike S2's reader C,
in plain torch).  Three build stages:

  solver-text-crops  (`crops`, `synth`)  the crops it learns from and is scored on: words at Terra's boxes on the 2000
                     drawing's three scans and non-text windows, each with the quarters and the answered blocks its
                     footprint touches; the owner's answered labels at Terra's boxes; the template reader's font and
                     prior learned from the training labels; and each quarter's synthetic code crops drawn with them
  solver-text-train  (`train`)  four readers and four 0/180 classifiers, one per held-out quarter, the answered
                     blocks held out of all four
  solver-text-read   (`read`)  every word box (`solver-boxes`) read both ways along its axis by its quarter's reader,
                     the posterior over the code list's printed forms, "other" and "none"; the measures

Folders (`store`); the network (`model`); the CTC sums and the direction mixture (`ctc`, the reading stage's only).
"""
