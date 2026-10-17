#!/usr/bin/env bash
# Build the WebAssembly viewer into dist/ (static site, e.g. for Netlify).
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --bin cartpole18
rm -rf dist && mkdir dist
cp web/index.html web/mq_js_bundle.js dist/
cp target/wasm32-unknown-unknown/release/cartpole18.wasm dist/
