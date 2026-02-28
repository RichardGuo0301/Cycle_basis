# Methodology: Guided Eulerian Splicing for Dynamic Path Recovery

## A. Problem Formulation: Algebraic vs. Sequential Constraints

Let $G = (V, E)$ be a directed graph modeling the physical layout, and let the initial periodic monitoring strategy be codified as an Eulerian circuit $\mathcal{W}_{orig}$, which maps to an optimal steady-state flow distribution $\mathbf{f}_{orig} \in \mathbb{Z}_{\ge 0}^{|E|}$. Upon the detection of environmental disruptions (e.g., edge or node blockages), the upper-level optimization scheme yields a corrective flow vector $\mathbf{C}_m \in \mathsf{nullspace}(\mathbf{B})$, where $\mathbf{B}$ is the incidence matrix of $G$. The updated optimal flow is given by:

$$ \mathbf{f}_{corr} = \mathbf{f}_{orig} + \mathbf{C}_m $$

By definition, $\mathbf{B}\mathbf{f}_{corr} = \mathbf{0}$, ensuring node-degree balance. Consequently, the updated multigraph $G_{corr}$ formed by the elements of $\mathbf{f}_{corr}$ is theoretically Eulerian. 

However, mapping the algebraic sum $\mathbf{f}_{orig} + \mathbf{C}_m$ back to a physically executable, continuous path $\mathcal{W}_{corr}$ introduces a critical non-triviality. The corrective vector $\mathbf{C}_m$ exhibits both positive entries (representing additive detours) and negative entries (representing subtractive route cancellations or shortcuts). Conventional local replanning methods (e.g., inserting shortest-path detours at the point of failure) inherently operate additively; they fail to structurally eliminate traversals ordered by $\mathbf{f}_{orig}$ when $(\mathbf{C}_m)_i < 0$. Attempting to dynamically splice local segments without global topological awareness frequently results in discontinuous paths or infinite loops.

## B. Guided Eulerian Splicing (GES) Algorithm

To resolve the discrepancy between the algebraic state space and sequential execution constraints, we propose the **Guided Eulerian Splicing (GES)** algorithm. GES abandons the paradigm of piecemeal local patching. Instead, it fully instantiates the ground-truth multigraph $G_{corr}$ given by $\mathbf{f}_{corr}$, and performs a constructive graph traversal that is heuristically biased by the original route $\mathcal{W}_{orig}$. 

This approach guarantees topological continuity while maximizing behavioral consistency with the pre-failure routine. The execution sequence $\mathcal{W}_{corr}$ is generated via a state-dependent transition function. Let $v_{curr} \in V$ be the current spatial state of the robot, and let $E_{avail}(v_{curr})$ be the set of outgoing edges from $v_{curr}$ that possess non-zero residual capacity in $G_{corr}$. The transition decision prioritizes edges based on three axiomatic rules:

### 1. Habitual Compliance (MATCH)
The algorithm inspects the next ordered edge $e_{orig} = (v_{curr}, v_{next})$ prescribed by the original sequence $\mathcal{W}_{orig}$. 
If $e_{orig} \in E_{avail}(v_{curr})$, the agent executes the edge. The capacity in $G_{corr}$ is decremented, and the pointer for $\mathcal{W}_{orig}$ advances.
$$ e_{t+1} = e_{orig} \quad \text{if} \quad e_{orig} \in E_{avail}(v_{curr}) $$
This ensures maximum overlap with the nominal traversal order, mitigating drastic temporal phase shifts in the resultant behavior.

### 2. Constructive Detouring (DETOUR)
If $e_{orig} \notin E_{avail}(v_{curr})$—either because the edge is physically obstructed ($e_{orig} \in E_{blocked}$) or analytically eliminated by the optimization ($(\mathbf{C}_m)_{e_{orig}} < 0$)—the agent must deviate.
The algorithm selects an alternative topological progression $e_{alt} \in E_{avail}(v_{curr})$ that is not a bridge in the residual graph of $G_{corr}$ (enforcing Fleury’s strict parity condition for Eulerian circuits). 
$$ e_{t+1} = \underset{e \in E_{avail}(v_{curr}) \setminus E_{bridge}}{\text{arg select}}(e) $$
This step physically manifests the positive components of the correction vector $(\mathbf{C}_m)_+ > 0$, guiding the agent along the newly synthesized mathematical detours.

### 3. Subtractive Cancellation (SKIP)
As the agent enacts a *DETOUR*, it transiently diverges from $\mathcal{W}_{orig}$. Because $G_{corr}$ conservatively embeds the negative components $(\mathbf{C}_m)_- < 0$ as algebraic cancellations, the newly traversed detour structurally bypasses obsolete segments of the original route. 
To maintain synchronization, the algorithm dynamically fast-forwards the structural pointer of $\mathcal{W}_{orig}$, skipping instructions until the spatial coordinate of the guide sequence re-aligns with the robot's physical location $v_{curr}$. Let $P_{skip}$ denote the sequence of skipped instructions:
$$ P_{skip} = \{ e \in \mathcal{W}_{orig} \mid f_{corr}(e) \text{ was exhausted by cancellation } \} $$
This allows GES to seamlessly digest negative algebraic flows without requiring complex backwards surgery on the physical path sequence.

## C. Theoretical Guarantees

**Theorem 1 (Completeness and Continuity):** 
Given a strongly connected multigraph $G_{corr}$ derived from an optimal, degree-balanced flow $\mathbf{f}_{corr}$, the Guided Eulerian Splicing algorithm produces an unbroken spatial sequence $\mathcal{W}_{corr}$ that perfectly traverses every edge in $G_{corr}$ exactly $f_{corr}(e)$ times, guaranteeing structural compliance with the MIQP solution.

*Proof sketch:* By fully instantiating $G_{corr}$ from $\mathbf{f}_{corr}$ and imposing Fleury’s bridge-avoidance constraint during *DETOUR* selections, the algorithm conforms to the classic sufficient conditions for Eulerian circuit extraction. The utilization of $\mathcal{W}_{orig}$ merely acts as a valid heuristic for selection bias without violating parity requirements, ensuring termination and topological soundness. 
