# Contributing

Thanks for looking at this repo. Three rules keep it what it is:

1. **Offline-first, zero dependencies.** The test suite must stay green with
   exactly `python -m unittest discover` on Python 3.10+ -- stdlib only. Anything that needs a key,
   the network, or a GPU stays lazy and optional (and money-gated behind
   owner approval; see CONTINUE.md).
2. **Honest claims.** Never state benchmark numbers that were not run. Stub
   and fixture output is labeled as mechanism validation, never as results.
   Negative and inconclusive findings get written down, not smoothed over.
3. **Independence.** Where this repo implements a paper, cite the paper
   first and never imply affiliation with its authors.

Practical notes:

- Commits: English, conventional style (`feat:`, `fix:`, `docs:`, `ci:`,
  `test:`, `chore:`).
- Before pushing, run the full offline suite and make sure it is green on
  your machine; CI runs the same command on Linux, Windows and macOS.
- Small, focused PRs with a clear rationale beat large rewrites. For
  anything that changes public behavior, open an issue first.

By contributing you agree that your contributions are licensed under the
repository's MIT license.
