# Deployment

The release repository is `Rtwotwo/DriveCoFi`. The static project page lives in
`site/`; `.github/workflows/pages.yml` deploys it when `main` is updated.

In repository **Settings → Pages**, select **GitHub Actions** as the source.
After the workflow succeeds, check the project page at
https://rtwotwo.github.io/DriveCoFi/ on desktop and mobile.

Before subsequent releases, run `python tools/check_site.py`,
`python tools/check_release.py`, and `bash -n scripts/release/*.sh`.
Review the proposed Git diff for private data and large artifacts.
