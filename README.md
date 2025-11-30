# MA (reorganized)

Quick notes:

- Code is maintained in `src/` as before. For convenience there's a `ma/` package that exposes many modules from `src/` so you can import them as `from ma.reward_module import InfoGainReward`.
- Utility scripts were copied into `scripts/` (you can run them directly).
- The `dynamic_patcher` utilities were archived into `src/archive/dynamic_patcher/` to keep them safe.

Recommended quick start (use your conda env):

```fish
# from repository root
python -m rl        # original entrypoint (keeps using src/ modules)
# or run scripts explicitly
python scripts/build_faiss_txt.py
```

If you want a proper package layout for development, consider moving code from `src/` into `src/ma/` and updating imports; I can do that next if you want.
