# Contributing to hdlib-skills

Thanks for helping improve the hdlib agent skills! This guide explains how to
add or modify a skill so it stays useful and trustworthy.

## Repository scope

This repository contains **only** Agent Skills for the
[`cumbof/hdlib`](https://github.com/cumbof/hdlib) library. Everything else
(general-purpose Python knowledge, unrelated tooling, etc.) belongs elsewhere.

## Adding a skill

1. **Pick a narrow scope.** A good skill addresses one concrete need
   (e.g. "encode tabular data", "build a graph model"). If your skill could
   easily be split in two, split it.
2. **Create the directory:** `skills/<skill-name>/`.
3. **Write `SKILL.md`** with YAML frontmatter:

   ```markdown
   ---
   name: hdlib-<your-skill>
   description: Use when ... (one or two sentences explaining when the agent should reach for this skill).
   ---

   # Body (markdown)
   ```

4. **Keep the `description` action-oriented.** Agents only see the description
   by default. Start with "Use when..." or "Use this to...".
5. **Body conventions** &mdash; see [Skill style](#skill-style) below.

## Skill style

Every `SKILL.md` should have these sections (in this order):

1. **One-line summary** &mdash; what hdlib feature this skill covers.
2. **When to use this skill** &mdash; bullets describing the agent-facing trigger.
3. **Key concepts** &mdash; minimum theory the agent needs to apply the API
   correctly.
4. **API surface** &mdash; the public functions/classes covered, with full
   signatures and parameter semantics.
5. **Recipes / examples** &mdash; copy-pasteable code blocks. **Every code block
   must work against the current `cumbof/hdlib` `main`.**
6. **Common pitfalls** &mdash; bullet-pointed.
7. **See also** &mdash; cross-references to other skills in this repo.

Code blocks must be valid Python 3.11+ and import only modules listed in
`hdlib`'s `requirements.txt` (`numpy`, `scikit-learn`, `scipy`, `qiskit`,
`qiskit-aer`, `qiskit-ibm-runtime`, `qiskit_machine_learning`, `mthree`,
`tabulate`).

## Verifying changes

Before submitting a PR:

1. Re-read the source of the hdlib function/class you're documenting (in
   `cumbof/hdlib` `hdlib/`). Make sure every parameter, return value, and
   exception is faithful to the implementation.
2. Run the canonical example from your skill. If the example needs data, use
   a small sklearn dataset (e.g. `sklearn.datasets.load_iris`).
3. Add (or extend) a `test_<short_name>` function in
   [`test/smoke_test.py`](./test/smoke_test.py) that exercises the same
   code block. Append the function to the `tests` list at the bottom of
   that file and re-run:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install hdlib
   python test/smoke_test.py
   ```

   The last line must read `All N skill smoke tests passed.`.
4. Open the index (`skills/README.md`) and add an entry for new skills.

## Versioning

`hdlib` follows semantic versioning. When a new major or minor release of
`hdlib` lands:

1. Re-read each skill's "API surface" section against the new source.
2. Update any signatures that changed.
3. Add a note about the version delta in `skills/README.md` if behaviour
   changed.

## Code of conduct

Be kind. Be specific. If you spot an error in a skill, please open an issue
or a PR with the citation (file + line) so reviewers can verify quickly.
