### Task 2: Glyph templates and the noise model

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/glyphs.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** Task 1's training labels and crops.
- **Produces:**
  - **`glyphs.Font`:** one template per character (a probability of ink per template pixel), a character width, an
    advance, and a size.
  - **`glyphs.learn_font(labels, iterations) -> Font`:** hard EM. Lay each training string out with the current
    templates, align by the best pose, then re-estimate the templates from the aligned crops with line pixels
    unobserved. Initialise from a plain sans rendering, and say which.
  - **`glyphs.NoiseModel`** with **`fit(...)`** (flip rates by distance from the stroke edge) and
    **`log_lik(rendered, crop, observed) -> float`**.
  - **`glyphs.render(font, text, pose) -> array`.**

- [ ] **Step 1: Failing tests:**
  - `test_em_recovers_templates_from_synthetic_text` (synthetic crops drawn from known templates plus noise);
  - `test_unobserved_pixels_change_nothing`.
- [ ] **Step 2:** Implement; the tests pass. Real run on the training labels:
  - report the training log-likelihood per EM round;
  - write a picture of the learned 2000 glyphs (scratch, say what it shows);
  - try adding block numbers and street names as training text (spec); keep them if the held-back training
    likelihood improves.
- [ ] **Step 3:** Full suite. Commit.

