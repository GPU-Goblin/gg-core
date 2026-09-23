# Licensing

**This file is an explanation, not the license.** The license is the `LICENSE` file
(PolyForm Noncommercial License 1.0.0), and it is the only authoritative text. When in doubt,
the license text prevails, and legal advice is worth more than this page.

## In one sentence

**Free for research and any noncommercial use; the code can be read and modified by anyone; it is
not open source in the OSI sense.**

## What you can do

- **Read, run, modify and redistribute** the code, for any noncommercial purpose.
- **Use it in a laboratory, a university, a public research organization, a government body or a
  charitable organization**, regardless of how they are funded. The license names these
  organizations explicitly.
- **Use it personally**: study, experimentation, hobby projects.
- **Publish your modifications**, provided you pass the license on with them.

## What you cannot do

- **Use it commercially** without written permission from the copyright holder: sell the software,
  sell a service built on it, or include it in a paid product.
- **Relicense** the code under another license, or transfer the rights.

## Why this choice

The project is aimed at scientists who cannot afford a computing center. The license says the
same thing: **the use that motivates the project is free, and commercial exploitation is
negotiated.** It is therefore not an "open source" license as defined by the Open Source
Initiative, which requires no discrimination against fields of endeavor. The accurate term is
**source-available**.

## Contributions

See `CONTRIBUTING.md`. An accepted contribution is published under this same license.

## Commercial use

Contact the copyright holder, Nicolas Michel, through the [GPU-Goblin organization on GitHub](https://github.com/GPU-Goblin).
Nothing in this repository grants such a license.

## Third-party code

The PolyForm license covers the code of GPU Goblin, not the code of others it uses.

- **Dependencies** keep their own licenses (`docs/DEPENDENCIES.md` lists them; all are permissive).
- **`vendor/burn-tch`** (in gg-core) is a copy of the `burn-tch` crate of the Burn project, under
  **MIT OR Apache-2.0**, with a one-line change explained in its `GPU-GOBLIN.md`. That directory
  stays under its original license (`LICENSE-MIT`, `LICENSE-APACHE` in it), change included; it is
  not relicensed under PolyForm.
- **LibTorch**, used by the native CUDA engine, is not part of any repository: `install.sh --cuda`
  downloads it from PyTorch, under PyTorch's own license.
