# Brand assets

Drop your brand files here and the whole site picks them up. Nothing in this
folder is required for the site to build — these are override slots.

| File you add                | What it controls                          | How to enable |
|-----------------------------|-------------------------------------------|---------------|
| `logo.svg` (or `.png`)      | Logo shown in the top-left header         | Uncomment `theme.logo` in `mkdocs.yml` |
| `favicon.png`               | Browser tab icon                          | Uncomment `theme.favicon` in `mkdocs.yml` |
| `hero.png`, diagrams, etc.  | Images you reference from markdown pages  | `![alt](../assets/hero.png)` |

## Colours, fonts and icons

- **Colours** live in [`docs/stylesheets/extra.css`](../stylesheets/extra.css) —
  edit the `--rlb-*` variables at the top.
- **Fonts** are set under `theme.font` in `mkdocs.yml`.
- **Icons** are set under `theme.icon` in `mkdocs.yml` (use any
  [Material](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/)
  or FontAwesome icon name).

> This file is excluded from the published site (see `exclude_docs` in
> `mkdocs.yml`), so it is safe to keep maintenance notes here.
