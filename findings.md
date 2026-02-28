# Findings & Decisions

## Requirements
- Analyze `edge_block_delta.py` and all code it involves/calls.
- Keep analysis within current methodology; no from-scratch replanning.
- Explain with persistent monitoring / cycle basis context.

## Research Findings
- `edge_block_delta.py` is a monolithic experiment pipeline: graph generation -> MCB -> MILP BFS solution -> random edge blocking -> 2-stage MIQP correction -> Euler/path repair -> visualization.
- Critical local dependencies: `cycle_basis_demo.py`, `horton_gram_mcb.py`, `visualizer.py`.
- The script includes its own large set of helper functions (incidence/mapping, BFS lattice search, MIQP solve, path splicing).
- A prior session note indicates concern around Section 9 guided Fleury path and count-matching semantics.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Build function-level call-chain table from actual definitions/imports | Avoid hand-wavy analysis |
| Separate algorithmic stages by objective/constraints | Clarifies where behavior diverges |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Automated AST command failed due missing `python` | Continue with static shell-based mapping |

## Resources
- `/Users/richardguo/Desktop/IROS_main/edge_block_delta.py`
- `/Users/richardguo/Desktop/IROS_main/cycle_basis_demo.py`
- `/Users/richardguo/Desktop/IROS_main/horton_gram_mcb.py`
- `/Users/richardguo/Desktop/IROS_main/visualizer.py`

## Visual/Browser Findings
- Not applicable (no browser/image inspection in this turn).

## Research Findings (ICRA/RSS/IROS methodology writing)
- Time window interpreted as 2024-01-01 to 2026-02-26; conferences primarily analyzed from ICRA 2024/2025, RSS 2024, and IROS 2024/2025 metadata.
- Repeated methodology pattern across planning papers:
  1) Formal problem statement with constraints/objective
  2) Method decomposition into 2-4 modules (global planner + local tracker/replanner)
  3) Algorithmic details with at least one pseudocode/optimization formulation
  4) Complexity or runtime discussion (often real-time feasibility)
  5) Adaptation/robustness extension (uncertainty/noise/dynamics)
  6) Evaluation protocol with ablations and strong baselines
- Persistent monitoring / IPP papers emphasize: uncertainty surrogate, budget constraints, online updates, and deployment-level compute budgets.

## Resources (methodology sample set)
- https://openreview.net/pdf?id=f7Op01TROT
- https://arxiv.org/abs/2504.02184
- https://www.roboticsproceedings.org/rss20/p001.html
- https://www.roboticsproceedings.org/rss20/p036.html
- https://www.roboticsproceedings.org/rss20/p039.html
- https://dblp.org/rec/conf/icra/JakkalaA24
- https://dblp.org/rec/conf/iros/ZahradkaK25
- https://dblp.org/rec/conf/iros/YaoMDLSK24
- https://dblp.org/rec/conf/iros/ZhangB0CCZWKYHK24
- https://dblp.org/db/conf/iros/index
- https://2025.ieee-iros.org/
