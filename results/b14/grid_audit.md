# B14c item 1: grid integrity audit

Verdict: **PASS: no deviations**

- Runs audited: 30 (PPO plain x10, PPO+D4 x10, DDQN x10), T = 1,500,000; produced by `scripts_b14_audit.py` (read-only; no training, no FINAL access).
- Git commit in every identity file and every validation row: `284873fb140bf2e6d0be68a8743eb2251c960f20` (1 distinct); HEAD at audit time `284873fb140bf2e6d0be68a8743eb2251c960f20`.
- Ledger `data/v2/final_access.log`: 5 build / 0 eval entries.
- Config files vs Gate changelog row (ee): ppo_plain ok; ppo_d4 ok; ddqn ok; queue ok.
- Manifest checker re-run on every run (`check_job_outputs`: `check_manifest_file`, val ids inside the manifest, final model, retention, T): 30/30 ok.
- Resumes: identity `resumes` list empty and exactly one scheduler start in the run log for 30/30 runs.
- git_dirty False in identity and in all validation rows: 30/30; shaping on: 30/30.
- Validation evals at t = 0, 25k, ..., 1.5M (61 each): 30/30.

## Deviations

None.

## Per run

| run | status | t trained | resumes | starts | checker | commit | dirty | shaping | final model | sha256 of final model |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| grid_ppo_plain_s0 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `43f29c0df831e1e11b0dea64acccbbb59b65a600a0d3fe45d260d0e9dbe0ab9a` |
| grid_ppo_d4_s0 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `0f92e1aec02a7456dac4829a7949d5338a57cfc628683869beb01c04dcb7fe35` |
| grid_ppo_plain_s1 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `2d17774a05f65f2830ad078f725af5bc89c3d2cf4e50cde89f4aa98a382a8f8c` |
| grid_ppo_d4_s1 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `9f3f4b321348ba0c032d034ec0a8d8aada9af0411b626606c497ec6f2f23d075` |
| grid_ppo_plain_s2 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `2750e90b46abd89ae610d022ae9ee62ab337c64d150c8a1d004609a8d34e1dd9` |
| grid_ppo_d4_s2 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `3e78e586c118241a83f825799ba89b0f0031e15bdb3b3537be82d7d74baaf626` |
| grid_ppo_plain_s3 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `e13d3485af2b9600a5cb80a672f4e39e74fc036a89fc5ea5421c7c852249a131` |
| grid_ppo_d4_s3 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `918d74a6cbf4e980430dd600db5ed06841569fa297916dfed8ef928b393b1efa` |
| grid_ppo_plain_s4 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `3e03155f4d385824f502805a18a63fe54a4d54d8dff9f55c63c84af5c4c80c29` |
| grid_ppo_d4_s4 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `947854c2d3f30fcb6ff2b2b01233aabfe664e9cdf969ae8df84db33f34aac481` |
| grid_ppo_plain_s5 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `3610f71445ae9200ed0dcd2994628818445392f380755111f12a68e0c0eb09e3` |
| grid_ppo_d4_s5 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `b664fd1d5e4c542709806715b52d90743edaa8a12cb40ceaaadf96d52ed24bbe` |
| grid_ppo_plain_s6 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `681159e3b8ccd17209938281ec8acd654a4f76d5217dc97fce980f55aec7d15e` |
| grid_ppo_d4_s6 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `a68753a824e0ea800c99f1d4234714484f8f991400d5d7ef0486f3ed7a58c12e` |
| grid_ppo_plain_s7 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `0a796384a7833ec27e265a50550e56d77791376fa5ecd1922cc78696915150f3` |
| grid_ppo_d4_s7 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `56d9cd46ad3d57333f541bf9fa981212f4acc12b25379dcdbab46cb96297410b` |
| grid_ppo_plain_s8 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `e37ed7809c978e1001b8ab1657dfc89008b140f446c23e4b16c0b2549827f35c` |
| grid_ppo_d4_s8 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `86277c0dfdc637b5439a5c63ea188ecf381960fdba6ac3d11e85d6988f0e8a10` |
| grid_ppo_plain_s9 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `b10e64caa867258e70c5b55181447aaf1823b467400245ff149f5c19ae0c01e3` |
| grid_ppo_d4_s9 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.zip | `7203968ecf74e0624ea3cb5bf1660dda2bb9de85c75f8a996c231b0961097f96` |
| grid_ddqn_s0 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `eb0be3471a465d56e838d87186f3593d6c532a21a2f3714836450a1fbb3b171f` |
| grid_ddqn_s1 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `c188013f3c62b533259acb432d0a0252de1fb1ab61947b4c6db1b4a2bd40676a` |
| grid_ddqn_s2 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `9a851bc58595a9f6ed9dfc963cff048a1b9fc212c1a0c29218b4e268b694de1c` |
| grid_ddqn_s3 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `875c7b164df44991fb7ab72c4d1274767f999dcd4644601df59c9c25779226c3` |
| grid_ddqn_s4 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `bd348e2b1427504b496d9913b2158be5c3d7f8e1cf40e8a297df3abe25eedd78` |
| grid_ddqn_s5 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `50c49ff2c7201020d16fb813edbff8121fa01300386703d8f71f49930a41d675` |
| grid_ddqn_s6 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `1febe374001444f41098e2b6dd9f5c8110e2f59a4285930d2a9a00f19b77ce57` |
| grid_ddqn_s7 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `cd2b9b4038705090211802f1571baf3ca98dddafc751ea57b10621461e5979cf` |
| grid_ddqn_s8 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `17e6de7dd13881650ed3fe4ab69c0c8cf9398bd13af11908dd217d87c2692927` |
| grid_ddqn_s9 | finished | 1,500,000 | 0 | 1 | ok | `284873f` | False | on | final.pt | `fc90793b4e71a5afa55ab56423e585d3f4e5d19baa69f05e10ea48c07cb0522d` |
