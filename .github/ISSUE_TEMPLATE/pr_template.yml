Checklist:

- [ ] Make sure all tests pass on the rc branch. Also make sure that the [milestone](https://github.com/stes/neural_cl/milestones) related to this release is fully done. Move issues that wont make it into the release to the next milestone, then close the milestone.
- [ ] Head to [`cebra.__init__`](neural_cl/cebra_public/cebra/__init__.py) and make sure that the `__version__` is set correctly.
- [ ] [Create a PR](https://github.com/stes/neural_cl/compare) between rc and `main`
- [ ] Tag the PR with the `release` [label](https://github.com/stes/neural_cl/issues/labels).
- [ ] A [github action will be run ](https://github.com/stes/neural_cl/blob/main/.github/workflows/internal-release.yml) -- if it doesnt start, try removing and re-adding the release label (step 4).
- [ ] Carefully check that the release looks fine --- the version from the PR will be pushed to [`release-staging`](https://github.com/AdaptiveMotorControlLab/cebra-internal/tree/release-staging) and [`staging`](https://github.com/AdaptiveMotorControlLab/cebra-internal/tree/staging) in the [`cebra-internal`](https://github.com/AdaptiveMotorControlLab/cebra-internal) repo. *Note: If you update the PR, these version will not be automatically updated. Repeat step 4 or trigger a manual workflow run if you need to update the staging version*
- [ ] If all looks good, tests pass and the PR is reviewed, merge the PR **using rebase merging**.
- [ ] Delete the branch
- [ ] Checkout the updated `main` branch, `git tag v1.2.3` with the correct format (if alpha/beta tags are used in __version__, use `v1.2.3a4` or `v1.2.3b4`), and `git push v1.2.3` the tag.
- [ ] Pushing the tag will update the release in `cebra-internal`. The source tree will land on [`main`](https://github.com/AdaptiveMotorControlLab/cebra-internal), the pre-build wheel and source distribution on [`release`](https://github.com/AdaptiveMotorControlLab/cebra-internal/tree/release).
