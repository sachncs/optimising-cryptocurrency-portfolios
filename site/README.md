# CPS site

Premium product landing page for the
[Crypto Portfolio System](https://github.com/sachncs/optimising-cryptocurrency-portfolios)
Python framework.

## Stack

- [Astro](https://astro.build/) — static site generator
- [Tailwind CSS](https://tailwindcss.com/) (via `@tailwindcss/vite`) — utility styling
- Vanilla JS for tab switching, scroll reveals, mobile nav

## Develop

```bash
npm install
npm run dev          # http://localhost:4321
```

## Build

```bash
npm run build        # outputs to dist/
npm run preview      # serves dist/
```

## Deploy

Deployed to GitHub Pages via `.github/workflows/pages.yml`. Every push to
`master` rebuilds and publishes the site to
`https://sachncs.github.io/optimising-cryptocurrency-portfolios/`.

## Structure

```
site/
├── astro.config.mjs      # Astro config (base path, Tailwind plugin)
├── public/               # Static assets (favicon, OG image, robots)
├── src/
│   ├── layouts/Base.astro
│   ├── pages/index.astro
│   ├── styles/global.css # Design system: tokens, components, animations
│   └── components/       # Hero, Pipeline, Features, Interfaces, Install, FAQ, CTA, Footer
└── package.json
```