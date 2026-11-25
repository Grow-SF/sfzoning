# Inputs of the street-and-block finder

Hand-made inputs of `sfzmaps find`; a person decides every row. The hand crops are in `../find-labels/`.

- `label_sources.csv` (`series,era,lots`): each series the finder reads, its era (the look of its print), and the lots
  that label it: `table` (the April 2000 table's lots), `year` (the lots of the series' own year), or empty (no lot
  labels; hand crops only).
- `unlabelled_lots.csv` (`blklot,name,why`): lots that carry no lot label on any map the lots label. A park held as
  one lot has roads inside it, and the lots cannot say where they run: labelling the whole lot "block" would teach the
  finder that the park's drives are block. Every pixel of a listed lot gets no label (no street or block label, no
  "no block edge" statement). The lot still bounds the streets beside it, and the block-edge cuts on its sides stay.
  Where the inside of such a lot should be taught, hand crops mark it (`../find-labels/`). The ids are the lots'
  `blklot`; `sfzmaps find labels` refuses a listed id that is not among a series' lots, naming the id and the series,
  so nobody trains believing a lot is switched off when it is not. Every grid and sheet label file records the
  list it was made under; after a change to this file, the label files made before it are refused until
  `sfzmaps find labels` makes them again.
