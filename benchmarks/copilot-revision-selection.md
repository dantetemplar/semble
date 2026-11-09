# Can the existing Copilot repository index select a historical commit?

No working selector was found in the inspected client or the bounded API probes
below. This is not proof that no undocumented backend capability exists.

## Source inspection

Inspected VS Code commit `00806ec273c665ede838f2f92f2c6b3fd007cb97` and
`@vscode/copilot-api` 0.5.2. The downloaded `githubCodeSearchService.ts` blob was
verified against that commit (`bb9bdb69b151dfc2effa4ab625dc698e2cac9d64`).

The [semantic search request](https://github.com/microsoft/vscode/blob/00806ec273c665ede838f2f92f2c6b3fd007cb97/extensions/copilot/src/platform/remoteCodeSearch/common/githubCodeSearchService.ts#L283-L327)
sends `scoping_query`, `prompt`, `include_embeddings`, `limit`, and
`embedding_model`. It does not send a revision selector. `location.commit_sha`,
`location.ref_name`, and the index's `semantic_commit_sha` describe returned/indexed
content; the client does not use them to request a historical snapshot.

For a different checkout, the [workspace search implementation](https://github.com/microsoft/vscode/blob/00806ec273c665ede838f2f92f2c6b3fd007cb97/extensions/copilot/src/platform/workspaceChunkSearch/node/codeSearch/codeSearchChunkSearch.ts#L640-L724)
searches the remote index and changed workspace files separately, removes remote
hits for changed files, and merges results. This is not historical search of the
existing remote index. Reproducing that pipeline would be a different baseline.

[GitHub's indexing documentation](https://docs.github.com/en/copilot/concepts/prompting-and-context/repository-indexing)
does not document an arbitrary-commit semantic-search API or guarantee that all
historical commits are retained.

## Read-only API experiment

Repository: `psf/requests`. The benchmark-pinned commit
`ef439eb779c1eba7cbdeeeb302b11e1e061b4b7d` was verified to exist through the commit
API. The existing semantic index was at
`3c7baaa05ab4aa49cc819ce3a819d260de19c157`, `refs/heads/main`.

All search requests used the same query, `HTTP session handling`, the Metis model,
and `limit=1`. The indexed SHA was unchanged before and after all 18 probes.
No index trigger, default-branch modification, or fork creation was performed.

| Probe | Result |
| --- | --- |
| Baseline request | HTTP 200, indexed SHA above |
| JSON body `commit_sha`, `ref`, `ref_name`, `revision`, or `indexed_commit_sha` set to the pinned SHA | HTTP 200, but every response and hit still used the existing indexed SHA |
| JSON body `commit_sha` set to 40 zeros or `ref` set to a nonexistent ref | Also HTTP 200 with the existing indexed SHA: these fields did not enforce revision selection |
| JSON body `ref_name=refs/tags/v2.31.0` (an existing tag) | Still returned `refs/heads/main` and the existing indexed SHA |
| Search URL `?ref=<pinned SHA>` or `?ref=<40 zeros>` | HTTP 200, existing indexed SHA |
| Scope `repo:psf/requests@<SHA>` with either pinned **or current indexed SHA**, or `@main` | HTTP 404, `repo_not_found` |
| Scope `repo:psf/requests ref:<SHA>` with either pinned or current indexed SHA | HTTP 404, `repo_not_found` |
| Index status URL `?ref=<pinned SHA>` or `?commit_sha=<pinned SHA>` | Reported the same existing semantic index |

The important negative control is the nonexistent ref/SHA: **HTTP 200 does not
mean a requested revision was honored**. Always check the actual response SHA
and every hit's commit, not just request success.

## Benchmark implications

- `exact` must reject mismatches. The preflight found exact matches for only 3 of
  the 63 benchmark repositories; 59 had an already-ready semantic index.
- `remote-index` can run the original queries, but must be reported separately
  with each actual indexed SHA and explicit availability/error counts.
- Pinning a dedicated repository's default branch and checking index/hit SHAs
  worked in the separate Loc-Bench experiment, but requires new indexing. If used
  here, benchmark copies should live in a dedicated organization, not clutter a
  personal account; no such copies have been created for this suite.
- Alternatively, rerun local methods at each existing indexed SHA. That changes
  the benchmark corpus and requires checking/revalidating the old annotations,
  especially line spans; it cannot be presented as the original pinned suite.
