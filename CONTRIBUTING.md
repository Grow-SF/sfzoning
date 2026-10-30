# Contributing

Thanks for helping. Bug reports, fixes, and work to adapt this project for
other cities are all welcome.

## Before your first pull request: the CLA

GrowSF asks every contributor to agree to the
[GrowSF Individual Contributor License Agreement](CLA.md), adapted from the
Apache Software Foundation's ICLA. You keep the copyright in your work; the
agreement gives GrowSF a license to use it and lets GrowSF keep the project's
licensing flexible.

To agree, post this comment on your first pull request, exactly as written:

> I have read the GrowSF Individual Contributor License Agreement (CLA.md,
> version 1.0) and I agree to it for all my present and future contributions
> to GrowSF.

Pull requests can't be merged until the author has posted it. If your employer
has rights in your work, make sure you're permitted to contribute (CLA
section 4); for contributions owned by a company, contact GrowSF.

## Licenses

By contributing, you agree that your contribution is distributed under the
project's licenses: code under AGPL-3.0-only (`LICENSE`), content and data
under CC BY-NC-SA 4.0 (`LICENSE-content`). See `NOTICE` for which is which.

If your contribution includes material you didn't write, such as code from
another project, data or images, say so in the pull request and give its
source and license.

## Practical notes

- `pipeline/`: Python 3.12 with uv. Run the tests with
  `cd pipeline && uv run pytest`. See `pipeline/README.md`.
- `site/`: Next.js. Run `npm test` and `npx playwright test` in `site/`.
  See `site/README.md`.
- Never commit secrets. The OpenAI key lives in `pipeline/.env`, which git
  ignores.
- Don't commit generated data (`pipeline/data/`). It's shared through the
  processed bundle instead.
