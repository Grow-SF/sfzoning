"""The zoning-code reader: for each printed zoning code on the 2000 sheets, a posterior over the codes from the scan's
own ink (glyph templates learned from the page, line pixels unobserved, a prior over codes).

  labels   the labels it reads (the prints, and the owner's answered code-check cases), the training set apart from
           the answers and their blocks, the code list, each label's crop straightened by its box's angle, and the
           long-line masks over it
  glyphs   the alphabet: glyph templates learned from the training labels by hard EM (from a plain sans), text
           rendered at a pose, and the noise from a rendered stroke to the scan's ink (line px unobserved)

Design: docs/superpowers/specs/2026-11-15-code-reader-design.md.
"""
