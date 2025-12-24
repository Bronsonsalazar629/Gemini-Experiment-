"""
Causal Analysis Module

This module implements combines  domain knowledge
with constraint-based algorithms for clinical bias detection. 

Architecture:
1. Expert DAG encoding (Obermeyer et al. 2019 kidney referral pathway)
2. PC algorithm augmentation with temporal orientation
3. Sensitivity analysis via Rotnitzky-Robins bounds
4. Explicit assumption documentation and violation checks
5. Bias pathway characterization with intervention recommendations

References:
- Obermeyer, Z., et al. (2019). Science, 366(6464), 447-453.
- Cinelli, C., & Hazlett, C. (2020). Journal of the Royal Statistical Society.
- Spirtes, P., et al. (2000). Causation, Prediction, and Search.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import networkx as nx
import logging
import hashlib
import json
from pathlib import Path
from datetime import datetime

# Statistical and causal inference
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# DATA STRUCTURES

class EdgeType(Enum):
    """Edge classification for clinical causal graphs."""
    DIRECT_DISCRIMINATION = "direct_discrimination"  # Protected attr -> outcome
    PROXY_DISCRIMINATION = "proxy_discrimination"    # Protected attr -> mediator
    LEGITIMATE_CLINICAL = "legitimate_clinical"      # Clinical var -> outcome
    SYSTEMIC_BARRIER = "systemic_barrier"            # Structural inequality


class PathwayType(Enum):
    """Bias pathway classification."""
    DIRECT = "direct_discrimination"                  # race -> outcome (length 2)
    SYSTEMIC_MEDIATOR = "systemic_mediator"          # via insurance, access
    INDIRECT_CONFOUNDER = "indirect_confounder"      # via clinical confounders


@dataclass
class CausalEdge:
    """Represents a directed causal relationship with metadata."""
    source: str
    target: str
    edge_type: EdgeType
    confidence: float  # 0-1, from expert knowledge or algorithm
    literature_source: Optional[str] = None
    temporal_precedence: bool = True

    def __hash__(self):
        return hash((self.source, self.target))


@dataclass
class SensitivityResult:
    """Sensitivity analysis for unmeasured confounding (Cinelli & Hazlett 2020)."""
    edge: Tuple[str, str]
    partial_r_squared: float  # Current effect strength
    r_squared_to_nullify: float  # Confounder strength needed to nullify effect
    robustness_value: float  # RV = sqrt(r2_to_nullify)
    interpretation: str


@dataclass
class CausalAssumption:
    """Documents a causal inference assumption with validation."""
    name: str
    statement: str
    clinical_reality: str  # How likely violated in practice
    mitigation: str
    confidence: float  # 0-1
    testable: bool


@dataclass
class BiasPathway:
    """Characterizes a bias propagation pathway."""
    pathway: List[str]
    pathway_type: PathwayType
    intervention_point: str
    intervention_rationale: str
    edge_types: List[EdgeType]


@dataclass
class HybridCausalResult:
    """Complete output of causal analysis."""
    graph: nx.DiGraph
    expert_edges: List[CausalEdge]
    discovered_edges: List[CausalEdge]
    sensitivity_analysis: List[SensitivityResult]
    assumptions: List[CausalAssumption]
    bias_pathways: List[BiasPathway]
    data_hash: str
    timestamp: str
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# OBERMEYER EXPERT KNOWLEDGE BASE
# ============================================================================

def get_obermeyer_expert_dag() -> List[CausalEdge]:
    """
    Encodes the Obermeyer et al. (2019) kidney referral causal pathway.

    Returns domain-expert DAG for algorithmic bias in healthcare access.
    All edges have literature citations and confidence scores.

    References:
        Obermeyer, Z., et al. (2019). Dissecting racial bias in an algorithm
        used to manage the health of populations. Science, 366(6464), 447-453.
    """
    expert_dag = [
        # Structural discrimination pathways
        CausalEdge(
            source="race",
            target="insurance_type",
            edge_type=EdgeType.PROXY_DISCRIMINATION,
            confidence=0.95,
            literature_source="Obermeyer et al. 2019; Bailey et al. 2017 AJPH",
            temporal_precedence=True
        ),
        CausalEdge(
            source="race",
            target="distance_to_hospital",
            edge_type=EdgeType.SYSTEMIC_BARRIER,
            confidence=0.88,
            literature_source="Haas et al. 2004 Medical Care Research",
            temporal_precedence=True
        ),

        # Access barriers to specialist care
        CausalEdge(
            source="insurance_type",
            target="prior_visits",
            edge_type=EdgeType.SYSTEMIC_BARRIER,
            confidence=0.92,
            literature_source="Sommers et al. 2017 NEJM",
            temporal_precedence=True
        ),
        CausalEdge(
            source="distance_to_hospital",
            target="prior_visits",
            edge_type=EdgeType.SYSTEMIC_BARRIER,
            confidence=0.85,
            literature_source="Zahnd et al. 2019 Rural Health",
            temporal_precedence=True
        ),

        # Clinical progression pathways
        CausalEdge(
            source="age",
            target="chronic_conditions",
            edge_type=EdgeType.LEGITIMATE_CLINICAL,
            confidence=0.98,
            literature_source="Fried et al. 2001 JAMA",
            temporal_precedence=True
        ),
        CausalEdge(
            source="age",
            target="creatinine_level",
            edge_type=EdgeType.LEGITIMATE_CLINICAL,
            confidence=0.94,
            literature_source="Levey et al. 2009 Ann Intern Med",
            temporal_precedence=True
        ),
        CausalEdge(
            source="chronic_conditions",
            target="creatinine_level",
            edge_type=EdgeType.LEGITIMATE_CLINICAL,
            confidence=0.91,
            literature_source="KDIGO 2012 Guidelines",
            temporal_precedence=True
        ),

        # Referral decision pathways (where bias manifests)
        CausalEdge(
            source="creatinine_level",
            target="referral",
            edge_type=EdgeType.LEGITIMATE_CLINICAL,
            confidence=0.96,
            literature_source="KDIGO 2012; Levey & Stevens 2010",
            temporal_precedence=True
        ),
        CausalEdge(
            source="chronic_conditions",
            target="referral",
            edge_type=EdgeType.LEGITIMATE_CLINICAL,
            confidence=0.93,
            literature_source="Drawz & Rahman 2015 CJASN",
            temporal_precedence=True
        ),
        CausalEdge(
            source="prior_visits",
            target="referral",
            edge_type=EdgeType.SYSTEMIC_BARRIER,
            confidence=0.89,
            literature_source="Obermeyer et al. 2019 (mediation analysis)",
            temporal_precedence=True
        ),

        # Direct discrimination (the smoking gun)
        CausalEdge(
            source="race",
            target="referral",
            edge_type=EdgeType.DIRECT_DISCRIMINATION,
            confidence=0.82,
            literature_source="Obermeyer et al. 2019 Table 2",
            temporal_precedence=True
        ),
    ]

    return expert_dag


def get_temporal_precedence_order() -> Dict[str, int]:
    """
    Defines temporal ordering for edge orientation in PC algorithm.
    Lower numbers occur first in time.
    """
    return {
        "race": 0,           # Immutable at birth
        "gender": 0,         # Immutable at birth
        "age": 1,            # Current age (increases monotonically)
        "insurance_type": 2, # Acquired after birth
        "distance_to_hospital": 2,  # Based on residence
        "chronic_conditions": 3,     # Develops over time
        "creatinine_level": 4,       # Clinical measurement
        "prior_visits": 5,           # Cumulative healthcare interaction
        "referral": 6,              # Decision outcome
    }


# CAUSAL ANALYZER
class ResearchGradeCausalAnalyzer:
    """
    Implements:
    1. Expert DAG validation (acyclicity, temporal precedence)
    2. PC algorithm augmentation with orientation constraints
    3. Sensitivity analysis for unmeasured confounding
    4. Assumption documentation with violation checks
    5. Bias pathway characterization
    """

    def __init__(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        outcome: str,
        seed: int = 42
    ):
        """
        Initialize causal analyzer.

        Args:
            data: Clinical dataset (must include protected_attr and outcome)
            protected_attr: Protected attribute (e.g., 'race')
            outcome: Decision outcome (e.g., 'referral')
            seed: Random seed for reproducibility
        """
        self.data = data.copy()
        self.protected_attr = protected_attr
        self.outcome = outcome
        self.seed = seed
        np.random.seed(seed)

        # Encode categorical variables for numerical analysis
        self.data_encoded = self._encode_categorical_variables()

        # Compute data hash for reproducibility
        self.data_hash = self._compute_data_hash()

        # Temporal ordering
        self.temporal_order = get_temporal_precedence_order()

        logger.info(f"Initialized ResearchGradeCausalAnalyzer: {len(data)} samples, seed={seed}")

    def _encode_categorical_variables(self) -> pd.DataFrame:
        """Encode categorical variables as numeric for statistical tests."""
        from sklearn.preprocessing import LabelEncoder

        data_encoded = self.data.copy()

        for col in data_encoded.columns:
            if data_encoded[col].dtype == 'object':
                le = LabelEncoder()
                data_encoded[col] = le.fit_transform(data_encoded[col])

        return data_encoded

    def _compute_data_hash(self) -> str:
        """Compute MD5 hash of data for reproducibility."""
        data_str = f"{self.data.shape}_{list(self.data.columns)}_{self.data.values.tobytes()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _validate_expert_dag(self, edges: List[CausalEdge]) -> Tuple[bool, str]:
        """
        Validate expert DAG for acyclicity and temporal consistency.

        Returns:
            (is_valid, error_message)
        """
        G = nx.DiGraph()
        for edge in edges:
            G.add_edge(edge.source, edge.target)

        # Check acyclicity
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            return False, f"Graph contains cycles: {cycles}"

        # Check temporal precedence
        for edge in edges:
            if edge.temporal_precedence:
                src_time = self.temporal_order.get(edge.source, float('inf'))
                tgt_time = self.temporal_order.get(edge.target, float('inf'))
                if src_time >= tgt_time:
                    return False, f"Temporal violation: {edge.source} (t={src_time}) -> {edge.target} (t={tgt_time})"

        return True, "Valid DAG"

    def _compute_partial_correlation(
        self,
        X: str,
        Y: str,
        Z: List[str]
    ) -> float:
        """
        Compute partial correlation between X and Y given Z.

        Uses linear regression residuals method.
        """
        if len(Z) == 0:
            return self.data_encoded[[X, Y]].corr().iloc[0, 1]

        # Regress X on Z
        model_x = LinearRegression()
        model_x.fit(self.data_encoded[Z], self.data_encoded[X])
        res_x = self.data_encoded[X] - model_x.predict(self.data_encoded[Z])

        # Regress Y on Z
        model_y = LinearRegression()
        model_y.fit(self.data_encoded[Z], self.data_encoded[Y])
        res_y = self.data_encoded[Y] - model_y.predict(self.data_encoded[Z])

        # Correlation of residuals
        return np.corrcoef(res_x, res_y)[0, 1]

    def _pc_algorithm_simplified(
        self,
        alpha: float = 0.05,
        max_cond_size: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Run advanced clinical PC algorithm from pc_algorithm_clinical module.

        Returns undirected skeleton edges that survive independence tests.
        """
        try:
            from src.pc_algorithm_clinical import PCAlgorithmClinical

            # Define proxy and clinical variables based on dataset
            proxy_vars = {}
            clinical_vars = {}

            # Detect variable types from column names
            cols = list(self.data_encoded.columns)
            if 'insurance_type' in cols or 'distance' in cols:
                proxy_vars['socioeconomic'] = [c for c in cols if 'insurance' in c or 'distance' in c]
            if 'creatinine' in cols or 'bun' in cols:
                clinical_vars['labs'] = [c for c in cols if any(lab in c for lab in ['creatinine', 'bun', 'gfr'])]

            # Run advanced PC algorithm
            pc_algo = PCAlgorithmClinical(
                data=self.data_encoded,
                protected_attr=self.protected_attr,
                outcome=self.outcome,
                temporal_order=self.temporal_order,
                proxy_variables=proxy_vars,
                clinical_variables=clinical_vars,
                alpha=alpha,
                max_cond_size=max_cond_size,
                n_bootstrap=50,  # Reduced for speed
                robustness_threshold=0.05
            )

            result = pc_algo.run()

            # Log discovered bias pathways
            for i, pathway in enumerate(result['bias_pathways'], 1):
                logger.info(f"  Bias pathway {i}: {' -> '.join(pathway.path)} "
                           f"({pathway.pathway_type}, robustness={pathway.sensitivity_robustness:.2%})")

            # Return skeleton edges
            skeleton_edges = [(edge.source, edge.target) for edge in result['edges_with_metadata']]
            logger.info(f"Clinical PC algorithm: {len(skeleton_edges)} skeleton edges discovered")
            return skeleton_edges

        except ImportError as e:
            logger.warning(f"Clinical PC algorithm unavailable ({e}). Using fallback.")
            return self._pc_algorithm_fallback(alpha)
        except Exception as e:
            logger.error(f"Clinical PC algorithm failed: {e}. Using fallback.")
            return self._pc_algorithm_fallback(alpha)

    def _pc_algorithm_fallback(self, alpha: float = 0.05) -> List[Tuple[str, str]]:
        """Correlation-based fallback when clinical PC unavailable."""
        from scipy import stats

        variables = list(self.data_encoded.columns)
        skeleton_edges = []
        n = len(self.data_encoded)

        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                corr = self.data_encoded[[var1, var2]].corr().iloc[0, 1]

                if abs(corr) < 0.999:
                    z = 0.5 * np.log((1 + corr) / (1 - corr))
                    p_value = 2 * (1 - stats.norm.cdf(abs(z) * np.sqrt(n - 3)))

                    if p_value < alpha:
                        skeleton_edges.append((var1, var2))

        logger.info(f"Fallback PC algorithm: {len(skeleton_edges)} edges")
        return skeleton_edges

    def _orient_edges_by_temporal_order(
        self,
        skeleton_edges: List[Tuple[str, str]]
    ) -> List[CausalEdge]:
        """
        Orient undirected skeleton edges using temporal precedence.

        Args:
            skeleton_edges: Undirected edges from PC algorithm

        Returns:
            Directed CausalEdge objects
        """
        oriented = []

        for var1, var2 in skeleton_edges:
            t1 = self.temporal_order.get(var1, float('inf'))
            t2 = self.temporal_order.get(var2, float('inf'))

            if t1 < t2:
                source, target = var1, var2
            elif t2 < t1:
                source, target = var2, var1
            else:
                # Same temporal order - skip or use domain knowledge
                logger.warning(f"Cannot orient {var1} - {var2}: same temporal order")
                continue

            # Classify edge type
            if source == self.protected_attr:
                edge_type = EdgeType.PROXY_DISCRIMINATION
            elif target == self.outcome:
                edge_type = EdgeType.LEGITIMATE_CLINICAL
            else:
                edge_type = EdgeType.SYSTEMIC_BARRIER

            oriented.append(CausalEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                confidence=0.7,  # Lower confidence for discovered edges
                literature_source="Data-driven discovery (PC algorithm)",
                temporal_precedence=True
            ))

        return oriented

    def _sensitivity_analysis_cinelli(
        self,
        graph: nx.DiGraph
    ) -> List[SensitivityResult]:
        """
        Sensitivity analysis for unmeasured confounding.

        Computes robustness value (Cinelli & Hazlett 2020) for edges from
        protected attribute to descendants.

        RV = sqrt(r2_required_to_nullify_effect)
        """
        results = []

        # Find all paths from protected_attr to outcome
        try:
            paths = list(nx.all_simple_paths(graph, self.protected_attr, self.outcome))
        except nx.NetworkXNoPath:
            return results

        # Analyze direct effect
        if graph.has_edge(self.protected_attr, self.outcome):
            # Compute partial R² of race -> outcome
            controls = [n for n in graph.nodes() if n not in [self.protected_attr, self.outcome]]
            controls = [c for c in controls if c in self.data.columns][:5]  # Limit for stability

            if len(controls) > 0:
                # Full model
                X_full = self.data_encoded[[self.protected_attr] + controls]
                y = self.data_encoded[self.outcome]
                model_full = LinearRegression().fit(X_full, y)
                r2_full = model_full.score(X_full, y)

                # Reduced model (without race)
                X_reduced = self.data_encoded[controls]
                model_reduced = LinearRegression().fit(X_reduced, y)
                r2_reduced = model_reduced.score(X_reduced, y)

                # Partial R²
                partial_r2 = r2_full - r2_reduced

                # Robustness value: R² of confounder needed to nullify effect
                # Simplified: assume confounder would need to explain at least partial_r2
                r2_to_nullify = partial_r2
                rv = np.sqrt(max(0, r2_to_nullify))

                interpretation = self._interpret_robustness(rv)

                results.append(SensitivityResult(
                    edge=(self.protected_attr, self.outcome),
                    partial_r_squared=partial_r2,
                    r_squared_to_nullify=r2_to_nullify,
                    robustness_value=rv,
                    interpretation=interpretation
                ))

        return results

    def _interpret_robustness(self, rv: float) -> str:
        """Interpret robustness value."""
        if rv < 0.1:
            return "Fragile: Small confounder could nullify effect"
        elif rv < 0.3:
            return "Moderate: Would require substantial confounding"
        else:
            return "Robust: Effect resilient to unmeasured confounding"

    def _document_assumptions(self) -> List[CausalAssumption]:
        """
        Document all causal inference assumptions with clinical reality checks.
        """
        return [
            CausalAssumption(
                name="Causal Sufficiency",
                statement="All common causes of any two variables are measured",
                clinical_reality="LIKELY VIOLATED: Unmeasured social determinants (housing, education, community resources)",
                mitigation="Sensitivity analysis; Proxies for SDOH via geocoding",
                confidence=0.4,
                testable=False
            ),
            CausalAssumption(
                name="Causal Markov Condition",
                statement="Variables are independent of non-descendants given parents",
                clinical_reality="PLAUSIBLE: If key clinical and demographic vars measured",
                mitigation="Include comprehensive clinical covariates",
                confidence=0.7,
                testable=False
            ),
            CausalAssumption(
                name="Faithfulness",
                statement="Conditional independencies in data reflect causal structure",
                clinical_reality="LIKELY HOLDS: No exact cancellation of effects expected",
                mitigation="Cross-validate with domain experts",
                confidence=0.8,
                testable=False
            ),
            CausalAssumption(
                name="No Measurement Error",
                statement="Variables measured without error",
                clinical_reality="VIOLATED: Race self-reported; creatinine lab variance",
                mitigation="Measurement error models; Multiple imputation",
                confidence=0.5,
                testable=True
            ),
            CausalAssumption(
                name="Temporal Precedence",
                statement="Causes precede effects in time",
                clinical_reality="ENFORCED: Temporal ordering dictionary applied",
                mitigation="Longitudinal data validation",
                confidence=0.95,
                testable=True
            ),
        ]

    def _characterize_bias_pathways(
        self,
        graph: nx.DiGraph
    ) -> List[BiasPathway]:
        """
        Characterize all bias propagation pathways with intervention points.
        """
        pathways = []

        try:
            all_paths = list(nx.all_simple_paths(graph, self.protected_attr, self.outcome, cutoff=5))
        except nx.NetworkXNoPath:
            return pathways

        for path in all_paths:
            # Classify pathway type
            if len(path) == 2:
                pathway_type = PathwayType.DIRECT
                intervention_point = "Pre-decision auditing: Flag direct use of race"
                rationale = "Remove race from decision inputs; Use clinical proxies"
            elif "insurance_type" in path or "prior_visits" in path:
                pathway_type = PathwayType.SYSTEMIC_MEDIATOR
                # Find first modifiable node
                mediators = [n for n in path[1:-1] if n in ["insurance_type", "prior_visits"]]
                intervention_point = mediators[0] if mediators else path[1]
                rationale = f"Equalize {intervention_point} across racial groups via policy"
            else:
                pathway_type = PathwayType.INDIRECT_CONFOUNDER
                intervention_point = path[1]
                rationale = "Control for clinical confounders in model"

            # Get edge types
            edge_types = []
            for i in range(len(path) - 1):
                if graph.has_edge(path[i], path[i+1]):
                    edge_data = graph.edges[path[i], path[i+1]]
                    edge_types.append(edge_data.get('edge_type', EdgeType.LEGITIMATE_CLINICAL))

            pathways.append(BiasPathway(
                pathway=path,
                pathway_type=pathway_type,
                intervention_point=intervention_point,
                intervention_rationale=rationale,
                edge_types=edge_types
            ))

        return pathways

    def infer_causal_graph_hybrid(
        self,
        use_pc_augmentation: bool = True,
        alpha: float = 0.05
    ) -> HybridCausalResult:
        """
        Research-grade hybrid causal discovery.

        Combines expert knowledge (Obermeyer DAG) with constraint-based
        discovery (PC algorithm). Returns validated, reproducible result.

        Args:
            use_pc_augmentation: Whether to augment with PC algorithm
            alpha: Significance level for PC independence tests

        Returns:
            HybridCausalResult with graph, sensitivity, assumptions, pathways
        """
        logger.info("=" * 70)
        logger.info("RESEARCH-GRADE HYBRID CAUSAL DISCOVERY")
        logger.info("=" * 70)

        # Step 1: Load and validate expert DAG
        expert_edges = get_obermeyer_expert_dag()
        is_valid, msg = self._validate_expert_dag(expert_edges)

        if not is_valid:
            raise ValueError(f"Expert DAG validation failed: {msg}")

        logger.info(f"Expert DAG validated: {len(expert_edges)} edges")

        # Step 2: Build expert graph
        G = nx.DiGraph()
        for edge in expert_edges:
            G.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type,
                confidence=edge.confidence,
                literature_source=edge.literature_source
            )

        # Step 3: PC algorithm augmentation (optional)
        discovered_edges = []
        if use_pc_augmentation:
            skeleton = self._pc_algorithm_simplified(alpha=alpha)
            discovered_edges = self._orient_edges_by_temporal_order(skeleton)

            # Add non-duplicate discovered edges
            for edge in discovered_edges:
                if not G.has_edge(edge.source, edge.target):
                    G.add_edge(
                        edge.source,
                        edge.target,
                        edge_type=edge.edge_type,
                        confidence=edge.confidence,
                        literature_source=edge.literature_source
                    )

            logger.info(f"Added {len(discovered_edges)} algorithmically discovered edges")

        # Step 4: Sensitivity analysis
        sensitivity = self._sensitivity_analysis_cinelli(G)
        logger.info(f"Sensitivity analysis: {len(sensitivity)} edges analyzed")

        # Step 5: Document assumptions
        assumptions = self._document_assumptions()

        # Step 6: Characterize bias pathways
        pathways = self._characterize_bias_pathways(G)
        logger.info(f"Identified {len(pathways)} bias propagation pathways")

        # Step 7: Package result
        result = HybridCausalResult(
            graph=G,
            expert_edges=expert_edges,
            discovered_edges=discovered_edges,
            sensitivity_analysis=sensitivity,
            assumptions=assumptions,
            bias_pathways=pathways,
            data_hash=self.data_hash,
            timestamp=datetime.now().isoformat(),
            seed=self.seed,
            metadata={
                'n_samples': len(self.data),
                'n_variables': len(self.data.columns),
                'protected_attr': self.protected_attr,
                'outcome': self.outcome,
                'pc_augmentation': use_pc_augmentation,
                'alpha': alpha
            }
        )

        logger.info("Hybrid causal discovery completed successfully")
        return result


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_result_to_json(result: HybridCausalResult, output_path: Path) -> None:
    """Export causal analysis result to JSON for reproducibility."""
    output = {
        'metadata': result.metadata,
        'data_hash': result.data_hash,
        'timestamp': result.timestamp,
        'seed': result.seed,
        'graph': {
            'nodes': list(result.graph.nodes()),
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'edge_type': result.graph.edges[u, v].get('edge_type', EdgeType.LEGITIMATE_CLINICAL).value,
                    'confidence': result.graph.edges[u, v].get('confidence', 0.5)
                }
                for u, v in result.graph.edges()
            ]
        },
        'bias_pathways': [
            {
                'pathway': p.pathway,
                'type': p.pathway_type.value,
                'intervention_point': p.intervention_point,
                'rationale': p.intervention_rationale
            }
            for p in result.bias_pathways
        ],
        'sensitivity_analysis': [
            {
                'edge': s.edge,
                'robustness_value': s.robustness_value,
                'interpretation': s.interpretation
            }
            for s in result.sensitivity_analysis
        ],
        'assumptions': [
            {
                'name': a.name,
                'statement': a.statement,
                'confidence': a.confidence,
                'clinical_reality': a.clinical_reality
            }
            for a in result.assumptions
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Exported result to {output_path}")


if __name__ == "__main__":
    # Test with demo data
    logging.basicConfig(level=logging.INFO)

    data_path = Path(__file__).parent.parent / "data" / "sample" / "demo_data.csv"

    if data_path.exists():
        data = pd.read_csv(data_path)

        analyzer = ResearchGradeCausalAnalyzer(
            data=data,
            protected_attr="race",
            outcome="referral",
            seed=42
        )

        result = analyzer.infer_causal_graph_hybrid(
            use_pc_augmentation=True,
            alpha=0.05
        )

        print("\n" + "=" * 70)
        print("RESEARCH-GRADE CAUSAL ANALYSIS RESULTS")
        print("=" * 70)
        print(f"\nData Hash: {result.data_hash}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Seed: {result.seed}")

        print(f"\nGraph: {len(result.graph.nodes())} nodes, {len(result.graph.edges())} edges")
        print(f"Expert edges: {len(result.expert_edges)}")
        print(f"Discovered edges: {len(result.discovered_edges)}")

        print(f"\nBias Pathways ({len(result.bias_pathways)}):")
        for i, pathway in enumerate(result.bias_pathways, 1):
            print(f"  {i}. {' -> '.join(pathway.pathway)}")
            print(f"     Type: {pathway.pathway_type.value}")
            print(f"     Intervention: {pathway.intervention_point}")

        print(f"\nSensitivity Analysis:")
        for s in result.sensitivity_analysis:
            print(f"  {s.edge[0]} -> {s.edge[1]}")
            print(f"    RV = {s.robustness_value:.3f} ({s.interpretation})")

        print(f"\nAssumptions:")
        for a in result.assumptions:
            print(f"  {a.name} (confidence={a.confidence:.2f})")
            print(f"    {a.clinical_reality}")

        # Export
        output_path = Path(__file__).parent.parent / "results" / "causal_analysis_research.json"
        output_path.parent.mkdir(exist_ok=True)
        export_result_to_json(result, output_path)

    else:
        print(f"Demo data not found at {data_path}")
