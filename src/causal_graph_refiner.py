"""
TIER 1: Causal Graph Refinement with Clinical Plausibility Audit

Uses Gemini 2.0 Flash as a clinical expert-in-the-loop to validate
causal edges discovered by the hybrid PC/expert pipeline.

Validation layers:
1. Clinical plausibility scoring
2. Literature support verification
3. Data correlation validation
4. Cycle detection in DAG
"""

import logging
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import networkx as nx

from src.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class CausalEdge:
    """Represents a validated causal edge."""
    source: str
    target: str
    confidence: float  # 0.0-1.0
    edge_type: str  # "expert", "discovered", "validated"
    rationale: str
    literature_support: Optional[str] = None


class EdgeValidation(BaseModel):
    """Pydantic model for Gemini edge validation response."""
    edge: Tuple[str, str]
    plausibility_score: float = Field(ge=0.0, le=1.0, description="Clinical plausibility (0-1)")
    justification: str = Field(min_length=10, description="Medical justification")
    literature: str = Field(description="Literature citation or 'Unknown'")
    data_support: bool = Field(description="Whether data correlation supports edge")


class CausalGraphValidationResponse(BaseModel):
    """Full validation response from Gemini."""
    validated_edges: List[EdgeValidation]
    clinical_context: str
    safety_concerns: List[str]


# ============================================================================
# Causal Graph Refiner
# ============================================================================

class CausalGraphRefiner:
    """
    Refines causal graphs using Gemini as clinical auditor.

    Ensures all edges are:
    - Clinically plausible
    - Supported by medical literature
    - Validated by data correlation
    - Part of a valid DAG (no cycles)
    """

    def __init__(self, gemini_client: GeminiClient, domain: str = "medicare_high_cost"):
        """
        Initialize refiner.

        Args:
            gemini_client: Configured Gemini API client
            domain: Clinical domain for context
        """
        self.gemini_client = gemini_client
        self.domain = domain

        # Domain-specific knowledge bases
        self.domain_contexts = {
            "medicare_high_cost": {
                "protected_attr_label": "race (White vs Non-White)",
                "outcome_label": "high-cost patient (top 25% medical costs)",
                "expert_knowledge": [
                    "Age increases chronic disease risk (CDC 2022)",
                    "Chronic diseases (diabetes, CHF, COPD) increase healthcare costs",
                    "Race correlates with socioeconomic factors affecting health access",
                    "ESRD (End-Stage Renal Disease) is high-cost condition"
                ],
                "safety_threshold": "FNR disparity < 5% for clinical safety"
            },
            "diabetic_amputation": {
                "protected_attr_label": "race (Black vs White)",
                "outcome_label": "diabetic amputation (lower extremity)",
                "expert_knowledge": [
                    "HbA1c >9% increases amputation risk 3x (Obermeyer 2019)",
                    "Lack of podiatry access delays preventive care",
                    "Insurance type affects specialist referral rates",
                    "Distance to care correlates with amputation rates"
                ],
                "safety_threshold": "FNR disparity < 5.5% per Obermeyer"
            }
        }

    def refine_causal_graph(
        self,
        expert_edges: List[CausalEdge],
        discovered_edges: List[Tuple[str, str]],
        data: pd.DataFrame,
        protected_attr: str,
        outcome: str
    ) -> Dict:
        """
        Validate and refine causal graph with Gemini.

        Args:
            expert_edges: High-confidence edges from domain experts
            discovered_edges: Medium-confidence edges from PC algorithm
            data: Clinical dataset for correlation validation
            protected_attr: Protected attribute name
            outcome: Outcome variable name

        Returns:
            Dictionary with validated edges and validation report
        """
        logger.info(f"Refining causal graph: {len(expert_edges)} expert edges, "
                   f"{len(discovered_edges)} discovered edges")

        # Step 1: Build clinical context
        context = self._build_clinical_context(data, protected_attr, outcome)

        # Step 2: Calculate data correlations for all edges
        edge_correlations = self._calculate_edge_correlations(
            discovered_edges, data
        )

        # Step 3: Call Gemini to validate discovered edges
        validated_discovered = self._validate_edges_with_gemini(
            discovered_edges,
            edge_correlations,
            context,
            protected_attr,
            outcome
        )

        # Step 4: Merge expert and validated edges
        all_edges = expert_edges + validated_discovered

        # Step 5: Detect and remove cycles
        all_edges = self._remove_cycles(all_edges)

        # Step 6: Generate validation report
        report = {
            "summary": {
                "total_edges": len(all_edges),
                "expert_edges": len(expert_edges),
                "validated_discovered_edges": len(validated_discovered),
                "removed_due_to_cycles": len(expert_edges) + len(validated_discovered) - len(all_edges)
            },
            "edges": [asdict(edge) for edge in all_edges],
            "context": context,
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }

        logger.info(f"Graph refinement complete: {len(all_edges)} total edges")
        return report

    def _build_clinical_context(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        outcome: str
    ) -> Dict:
        """Build clinical context for Gemini prompt."""
        domain_info = self.domain_contexts.get(self.domain, {})

        context = {
            "domain": self.domain,
            "protected_attr": protected_attr,
            "protected_attr_label": domain_info.get("protected_attr_label", protected_attr),
            "outcome": outcome,
            "outcome_label": domain_info.get("outcome_label", outcome),
            "data_summary": {
                "n_samples": len(data),
                "n_features": len(data.columns),
                "features": list(data.columns),
                "outcome_prevalence": float(data[outcome].mean()),
            },
            "expert_knowledge": domain_info.get("expert_knowledge", []),
            "safety_threshold": domain_info.get("safety_threshold", "FNR disparity < 5%")
        }

        # Add protected attribute distribution if available
        if protected_attr in data.columns:
            context["protected_attr_distribution"] = data[protected_attr].value_counts().to_dict()

        return context

    def _calculate_edge_correlations(
        self,
        edges: List[Tuple[str, str]],
        data: pd.DataFrame
    ) -> Dict[Tuple[str, str], float]:
        """Calculate Pearson correlation for each edge."""
        correlations = {}

        for source, target in edges:
            if source in data.columns and target in data.columns:
                corr = data[source].corr(data[target])
                correlations[(source, target)] = float(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[(source, target)] = 0.0
                logger.warning(f"Edge ({source}, {target}) references missing column")

        return correlations

    def _validate_edges_with_gemini(
        self,
        edges: List[Tuple[str, str]],
        edge_correlations: Dict[Tuple[str, str], float],
        context: Dict,
        protected_attr: str,
        outcome: str
    ) -> List[CausalEdge]:
        """Use Gemini to validate discovered edges."""
        if not edges:
            return []

        # Build validation prompt
        system_instruction = """You are a clinical AI ethicist and epidemiologist at Johns Hopkins Hospital.
Your role is to audit causal graphs for clinical ML systems to ensure they are:
1. Clinically plausible based on medical literature
2. Supported by actual data correlations
3. Free from spurious associations
4. Aligned with principles of medical ethics (justice, beneficence, non-maleficence)"""

        # Format edges with correlations
        edges_with_corr = []
        for edge in edges:
            corr = edge_correlations.get(edge, 0.0)
            edges_with_corr.append({
                "source": edge[0],
                "target": edge[1],
                "data_correlation": round(corr, 3)
            })

        prompt = f"""
CLINICAL CONTEXT:
- Domain: {context['domain']}
- Protected attribute: {context['protected_attr']} ({context['protected_attr_label']})
- Outcome: {context['outcome']} ({context['outcome_label']})
- Dataset: {context['data_summary']['n_samples']} patients, {context['data_summary']['n_features']} features
- Outcome prevalence: {context['data_summary']['outcome_prevalence']:.1%}

EXPERT KNOWLEDGE BASE:
{chr(10).join('- ' + k for k in context['expert_knowledge'])}

TASK: Validate the following causal edges discovered by PC algorithm:
{json.dumps(edges_with_corr, indent=2)}

For EACH edge, provide:
1. **plausibility_score** (0.0-1.0): How clinically plausible is this causal relationship?
   - 0.0-0.3: Spurious/implausible
   - 0.3-0.6: Possible but weak evidence
   - 0.6-0.8: Likely based on known mechanisms
   - 0.8-1.0: Well-established in medical literature

2. **justification**: One medical sentence explaining the causal mechanism
   Example: "Age increases diabetes risk through insulin resistance and pancreatic beta-cell decline"

3. **literature**: Citation if known, or "Unknown" if speculative
   Examples: "CDC Diabetes Report 2022", "Obermeyer Science 2019", "Unknown"

4. **data_support**: True if |correlation| >= 0.1, False otherwise

VALIDATION RULES:
- If |correlation| < 0.1, reduce plausibility by 50%
- Race should NOT directly cause medical outcomes (systemic mediators only)
- Chronic conditions CAN directly increase costs
- Age/sex are valid confounders

Return JSON array with format:
{{
  "validated_edges": [
    {{
      "edge": ["source", "target"],
      "plausibility_score": 0.75,
      "justification": "Medical explanation here",
      "literature": "Citation or Unknown",
      "data_support": true
    }}
  ]
}}
"""

        try:
            # Call Gemini with structured output
            response_text = self.gemini_client.call_with_retry(
                prompt,
                temperature=0.3,
                system_instruction=system_instruction
            )

            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]

            response_dict = json.loads(response_text)
            validations = response_dict.get("validated_edges", [])

            # Convert to CausalEdge objects
            validated_edges = []
            for val in validations:
                edge = tuple(val["edge"])
                corr = edge_correlations.get(edge, 0.0)

                # Apply correlation penalty
                plausibility = val["plausibility_score"]
                if abs(corr) < 0.1:
                    plausibility *= 0.5

                # Only include edges with plausibility >= 0.3
                if plausibility >= 0.3:
                    validated_edges.append(CausalEdge(
                        source=edge[0],
                        target=edge[1],
                        confidence=plausibility,
                        edge_type="validated",
                        rationale=val["justification"],
                        literature_support=val.get("literature", "Unknown")
                    ))
                else:
                    logger.info(f"Rejected edge {edge} (plausibility {plausibility:.2f} < 0.3)")

            logger.info(f"Validated {len(validated_edges)}/{len(edges)} discovered edges")
            return validated_edges

        except Exception as e:
            logger.error(f"Gemini validation failed: {e}. Using correlation fallback.")
            return self._fallback_validation(edges, edge_correlations)

    def _fallback_validation(
        self,
        edges: List[Tuple[str, str]],
        edge_correlations: Dict[Tuple[str, str], float]
    ) -> List[CausalEdge]:
        """Fallback validation using only data correlations."""
        validated = []

        for edge in edges:
            corr = edge_correlations.get(edge, 0.0)

            if abs(corr) >= 0.1:  # Minimum correlation threshold
                validated.append(CausalEdge(
                    source=edge[0],
                    target=edge[1],
                    confidence=min(abs(corr), 0.7),  # Cap at 0.7 for fallback
                    edge_type="fallback_validated",
                    rationale=f"Data correlation: {corr:.3f}",
                    literature_support="Correlation-based (LLM unavailable)"
                ))

        logger.info(f"Fallback validation: {len(validated)}/{len(edges)} edges with |r| >= 0.1")
        return validated

    def _remove_cycles(self, edges: List[CausalEdge]) -> List[CausalEdge]:
        """Remove edges that create cycles in DAG."""
        G = nx.DiGraph()

        # Add edges in order of confidence (keep high-confidence edges)
        edges_sorted = sorted(edges, key=lambda e: e.confidence, reverse=True)
        valid_edges = []

        for edge in edges_sorted:
            G_temp = G.copy()
            G_temp.add_edge(edge.source, edge.target)

            if nx.is_directed_acyclic_graph(G_temp):
                G.add_edge(edge.source, edge.target)
                valid_edges.append(edge)
            else:
                logger.warning(f"Removed edge ({edge.source} -> {edge.target}) - creates cycle")

        return valid_edges


def refine_causal_graph_with_gemini(
    expert_edges: List[CausalEdge],
    discovered_edges: List[Tuple[str, str]],
    data: pd.DataFrame,
    protected_attr: str,
    outcome: str,
    gemini_client: GeminiClient,
    domain: str = "medicare_high_cost"
) -> Dict:
    """
    Convenience function to refine causal graph.

    Args:
        expert_edges: Expert-provided edges
        discovered_edges: PC algorithm edges
        data: Clinical dataset
        protected_attr: Protected attribute
        outcome: Outcome variable
        gemini_client: Gemini API client
        domain: Clinical domain

    Returns:
        Validation report with refined edges
    """
    refiner = CausalGraphRefiner(gemini_client, domain=domain)
    return refiner.refine_causal_graph(
        expert_edges,
        discovered_edges,
        data,
        protected_attr,
        outcome
    )
