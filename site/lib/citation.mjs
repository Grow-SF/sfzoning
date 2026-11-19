// DRAFT COPY — NOT REVIEWED BY THE OWNER. Citation wording is a first draft.
/**
 * Citation strings for a dated data release. Pure functions, shared by the site and the
 * publish script (Node), so this file must not import any .ts module.
 *
 * Plain text only: the page italicises the title itself.
 *
 * @typedef {{ url: string; apa: string; chicago: string; bibtex: string; cff: string }} Citations
 */

/** Same text as DATA_CREDIT in lib/credits.ts (a test keeps the two equal). */
const CREDIT = "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)";
const TITLE = "A History of San Francisco’s Zoning";
const AUTHOR_FAMILY = "Bacio";
const AUTHOR_GIVEN = "Steven";
const PUBLISHER = "GrowSF";
const LICENSE_TEXT = "CC BY-NC-SA 4.0";
const CFF_LICENSE = "CC-BY-NC-SA-4.0";

/**
 * Citations for one release.
 * @param {{ id: string; date: string }} release  id is the release id (e.g. "2026-11-18-2");
 *   date is the release date (YYYY-MM-DD), whose first four characters are the year.
 * @returns {Citations}
 */
export function citations({ id, date }) {
  const url = `https://sfzoning.com/data/releases/${id}`;
  const year = date.slice(0, 4);
  const apa = `${AUTHOR_FAMILY}, ${AUTHOR_GIVEN[0]}. (${year}). ${TITLE} (Version ${id}) [Data set]. ${PUBLISHER}. ${url}`;
  const chicago = `${AUTHOR_FAMILY}, ${AUTHOR_GIVEN}. ${year}. ${TITLE}. Data release ${id}. ${PUBLISHER}. ${url}.`;
  const bibtex = [
    `@misc{bacio_sfzoning_${year},`,
    `  author    = {${AUTHOR_FAMILY}, ${AUTHOR_GIVEN}},`,
    `  title     = {${TITLE}},`,
    `  year      = {${year}},`,
    `  version   = {${id}},`,
    `  publisher = {${PUBLISHER}},`,
    `  url       = {${url}},`,
    `  note      = {Data set, ${LICENSE_TEXT}}`,
    `}`,
  ].join("\n");
  const cff = [
    "cff-version: 1.2.0",
    `message: "If you use this data, please cite it. ${CREDIT}"`,
    "type: dataset",
    `title: "${TITLE}"`,
    "authors:",
    `  - family-names: ${AUTHOR_FAMILY}`,
    `    given-names: ${AUTHOR_GIVEN}`,
    "publisher:",
    `  name: ${PUBLISHER}`,
    `version: "${id}"`,
    `date-released: ${date}`,
    `url: "${url}"`,
    `license: ${CFF_LICENSE}`,
    "",
  ].join("\n");
  return { url, apa, chicago, bibtex, cff };
}
