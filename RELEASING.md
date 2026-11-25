# Releasing ml-pipes-vlm

`ml-pipes-vlm` is an independently versioned, opt-in community distribution
maintained in the [`vlm-tasks`](https://github.com/trained-by-humans/vlm-tasks)
repository. It uses the `ml_pipes.vlm` namespace and follows the
[verified community package policy](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/verified_community_packages.md).
Framework dependencies use `>=0.1.3,<0.2.0`; PydanticAI is pinned to the tested
API version, `2.61.0`.

The staging and promotion contract is shared with
[`ml-pipes-supervision`](https://github.com/trained-by-humans/ml-pipes-supervision/blob/main/RELEASING.md)
and
[`ml-pipes-ultralytics`](https://github.com/trained-by-humans/ml-pipes-ultralytics/blob/main/RELEASING.md).

## Prepare

Update `project.version`, dependency bounds, release notes, installation
instructions and affected tests together in a release preparation PR.

CI tests Python 3.11 through 3.13, including the local Transformers adapter
with fake weights. It runs the synthetic pipelines, builds wheel and source
artifacts, checks their contents and metadata, and installs the wheel in a clean
environment.
Require `Compat: Gate` and `Build and validate distribution` in branch rules.

## Stage on TestPyPI

After merging, create and push an annotated `v<project.version>` tag, such as
`v0.1.0`. The `Stage` workflow:

1. Checks that the tag matches the package version and runs CI.
2. Builds and validates the distributions and records SHA-256 checksums.
3. Attaches these artifacts to a GitHub Release.
4. Downloads and verifies those assets, publishes them to TestPyPI via OIDC,
   and verifies an installation and import. The package wheel is fetched only
   from TestPyPI; runtime dependencies are resolved from PyPI.

Configure the TestPyPI trusted publisher with:

```text
Owner:       trained-by-humans
Repository:  vlm-tasks
Workflow:    .github/workflows/stage-release.yml
Environment: test-pypi
```

TestPyPI cannot reuse uploaded filenames. Increment the candidate version
when rebuilding a candidate that has already been uploaded.

## Promote to PyPI

After staging succeeds, dispatch `Release` from the staged tag. It requires a
successful Stage run for the same commit, rejects branch refs, downloads the
existing release assets, verifies their checksums and publishes the same
artifacts to PyPI. Configure required reviewers on the `pypi` environment and
configure the PyPI trusted publisher with:

```text
Owner:       trained-by-humans
Repository:  vlm-tasks
Workflow:    .github/workflows/publish-release.yml
Environment: pypi
```

Organization administrators control release tags and deployment approval.
Protect `v*` tags and require review for changes to package metadata and release
workflows. Configure trusted publishers and environment protection on GitHub
before publishing.

## Distribution contents

Wheels contain only the `ml_pipes.vlm` package and distribution metadata.
Source distributions contain the same library source, `pyproject.toml`,
`README.md`, `LICENSE`, VCS ignore rules and generated package metadata. Both
retain the public `ml_pipes.vlm.validation` API and its caption-scoring license
notice.

Benchmarks are applications built on the task APIs and remain in the
repository, along with tests, examples, documentation and release tooling.
They are excluded from both PyPI artifacts. CI still runs the full repository
test suite, including benchmark tests. CI and staging check the actual archive
contents with `python tests/check_distribution.py dist` before publication.
