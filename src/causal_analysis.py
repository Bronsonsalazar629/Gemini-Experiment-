"""
Causal Analysis Module with Gemini 3 Integration

This module provides functionality for causal graph inference and causal pathway
analysis to detect bias in clinical ML models using LLM-enhanced causal reasoning.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import networkx as nx
import logging
import time
import json
import hashlib
from pathlib import Path
from google import genai
import yaml
import re

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_api_config() -> Dict[str, Any]:
    """Load API configuration from yaml file."""
    config_path = Path(__file__).parent.parent / "config" / "api_keys.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load API config: {e}")
        return {}


def _init_gemini() -> Optional[Any]:
    """Initialize Gemini API client with error handling."""
    config = _load_api_config()
    api_key = config.get('gemini_api_key')

    if not api_key or not config.get('enable_llm', False):
        logger.info("Gemini API not configured or disabled")
        return None

    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini API initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")
        return None


def _call_gemini_with_retry(
    client: Any,
    prompt: str,
    max_retries: int = 3
) -> Optional[str]:
    """Call Gemini API with exponential backoff retry logic."""
    config = _load_api_config()
    model_name = config.get('gemini_model', 'gemini-1.5-flash')

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
            logger.warning(f"Empty response from Gemini (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)

    logger.error("All Gemini API retry attempts failed")
    return None


def _get_cache_key(data: pd.DataFrame, sensitive_attr: str, outcome: str) -> str:
    """Generate cache key for causal graph results."""
    data_hash = hashlib.md5(
        f"{data.shape}_{list(data.columns)}_{sensitive_attr}_{outcome}".encode()
    ).hexdigest()
    return f"causal_graph_{data_hash}.json"


def _save_to_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Save results to cache."""
    try:
        cache_file = CACHE_DIR / cache_key
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def _load_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load results from cache."""
    try:
        cache_file = CACHE_DIR / cache_key
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
    return None


def _parse_dot_to_graph(dot_string: str) -> Optional[nx.DiGraph]:
    """Parse DOT format string to NetworkX DiGraph."""
    try:
        # Clean up the DOT string
        dot_string = dot_string.strip()

        # Remove markdown code blocks if present
        dot_string = re.sub(r'```(?:dot|graphviz)?\n?', '', dot_string)
        dot_string = re.sub(r'```', '', dot_string)

        # Extract edges from DOT format
        G = nx.DiGraph()

        # Find all edges (A -> B)
        edge_pattern = r'(\w+)\s*->\s*(\w+)'
        edges = re.findall(edge_pattern, dot_string)

        if edges:
            G.add_edges_from(edges)
            logger.info(f"Parsed {len(edges)} edges from DOT format")
            return G

        logger.warning("No edges found in DOT format")
        return None

    except Exception as e:
        logger.error(f"Failed to parse DOT format: {e}")
        return None


def _create_default_graph(data: pd.DataFrame, sensitive_attr: str, outcome: str) -> nx.DiGraph:
    """Create a default causal graph based on clinical domain knowledge."""
    G = nx.DiGraph()
    features = data.columns.tolist()
    G.add_nodes_from(features)

    # Add edges based on typical clinical causal relationships
    if sensitive_attr in features and outcome in features:
        G.add_edge(sensitive_attr, outcome, edge_type="direct")

    if 'age' in features:
        G.add_edge('age', outcome)
        if 'chronic_conditions' in features:
            G.add_edge('age', 'chronic_conditions')

    if 'chronic_conditions' in features and outcome in features:
        G.add_edge('chronic_conditions', outcome)

    if 'insurance_type' in features:
        G.add_edge(sensitive_attr, 'insurance_type', edge_type="proxy")
        G.add_edge('insurance_type', outcome)

    if 'creatinine_level' in features and outcome in features:
        G.add_edge('creatinine_level', outcome)

    logger.info(f"Created default graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G


class CausalAnalyzer:
    """
    Analyzes causal relationships in clinical data to identify bias pathways.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        outcome: str,
        known_graph: Optional[nx.DiGraph] = None
    ):
        """
        Initialize the causal analyzer.

        Args:
            data: Clinical dataset with features and outcome
            protected_attr: Protected attribute column name
            outcome: Outcome variable column name
            known_graph: Optional pre-specified causal graph
        """
        self.data = data
        self.protected_attr = protected_attr
        self.outcome = outcome
        self.causal_graph = known_graph
        self.gemini_client = _init_gemini()

    def infer_causal_graph(
        self,
        use_cache: bool = True,
        llm_enhanced: bool = True
    ) -> Dict[str, Any]:
        """
        Infer causal graph structure from data using Gemini 3.

        Args:
            use_cache: Whether to use cached results
            llm_enhanced: Whether to use LLM for graph refinement

        Returns:
            Dict with 'graph' (nx.DiGraph) and 'llm_explanation' (str)
        """
        # Check cache first
        if use_cache:
            cache_key = _get_cache_key(self.data, self.protected_attr, self.outcome)
            cached = _load_from_cache(cache_key)
            if cached:
                logger.info("Using cached causal graph")
                # Reconstruct graph from edge list
                graph = nx.DiGraph()
                graph.add_edges_from(cached['edges'])
                return {
                    'graph': graph,
                    'llm_explanation': cached['explanation']
                }

        # Create base graph
        base_graph = _create_default_graph(self.data, self.protected_attr, self.outcome)

        if not llm_enhanced or not self.gemini_client:
            explanation = self._generate_basic_explanation(base_graph)
            result = {'graph': base_graph, 'llm_explanation': explanation}
            self.causal_graph = base_graph

            # Cache result
            if use_cache:
                cache_data = {
                    'edges': list(base_graph.edges()),
                    'explanation': explanation
                }
                _save_to_cache(cache_key, cache_data)

            return result

        # Use Gemini to refine graph
        logger.info("Refining causal graph with Gemini 3")

        variables = list(self.data.columns)
        prompt = f"""You are a clinical AI ethicist and causal inference expert.

Variables in dataset: {variables}
Sensitive attribute: '{self.protected_attr}'
Outcome: '{self.outcome}'

Task: Refine the causal relationships using medical domain knowledge. Consider:
1. Direct pathways from {self.protected_attr} to {self.outcome}
2. Mediators (e.g., insurance_type, access to care)
3. Confounders (e.g., age, chronic conditions)
4. Clinically plausible relationships

Return ONLY valid DOT format (directed graph). Example:
digraph {{
    race -> insurance_type;
    insurance_type -> referral;
    age -> chronic_conditions;
    chronic_conditions -> referral;
}}

Return only the DOT format, no additional text."""

        response = _call_gemini_with_retry(self.gemini_client, prompt)

        if response:
            refined_graph = _parse_dot_to_graph(response)
            if refined_graph and len(refined_graph.edges()) > 0:
                # Generate explanation
                explanation_prompt = f"""Explain this causal graph for clinical bias analysis:

Variables: {variables}
Sensitive attribute: {self.protected_attr}
Outcome: {self.outcome}

Graph edges: {list(refined_graph.edges())}

Provide a 3-sentence explanation focusing on:
1. Key bias pathways
2. Mediating mechanisms
3. Clinical implications

Be concise and technical."""

                explanation = _call_gemini_with_retry(self.gemini_client, explanation_prompt)
                if not explanation:
                    explanation = self._generate_basic_explanation(refined_graph)

                self.causal_graph = refined_graph

                # Cache result
                if use_cache:
                    cache_data = {
                        'edges': list(refined_graph.edges()),
                        'explanation': explanation
                    }
                    _save_to_cache(cache_key, cache_data)

                return {'graph': refined_graph, 'llm_explanation': explanation}

        # Fallback to base graph
        logger.warning("Using fallback base graph")
        explanation = self._generate_basic_explanation(base_graph)
        self.causal_graph = base_graph

        return {'graph': base_graph, 'llm_explanation': explanation}

    def _generate_basic_explanation(self, graph: nx.DiGraph) -> str:
        """Generate basic explanation without LLM."""
        edges = list(graph.edges())
        explanation = f"""Causal graph analysis (template-based):

Identified {len(edges)} causal relationships in the clinical data.

Key pathways from {self.protected_attr} to {self.outcome}:
{chr(10).join(f"- {src} → {tgt}" for src, tgt in edges[:5])}

This graph represents potential bias propagation routes that should be examined for fairness violations."""

        return explanation

    def identify_bias_pathways(self) -> List[List[str]]:
        """
        Identify all causal pathways from protected attribute to outcome.

        Returns:
            List of pathways, where each pathway is a list of node names
        """
        if self.causal_graph is None:
            raise ValueError("Causal graph not initialized. Call infer_causal_graph() first.")

        try:
            all_paths = list(nx.all_simple_paths(
                self.causal_graph,
                source=self.protected_attr,
                target=self.outcome
            ))
            logger.info(f"Found {len(all_paths)} bias pathways")
            return all_paths
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning(f"No paths found from {self.protected_attr} to {self.outcome}: {e}")
            return []


def infer_causal_graph(
    data_df: pd.DataFrame,
    protected_attr: str,
    outcome: str,
    use_cache: bool = True,
    llm_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to infer causal graph from data.

    Args:
        data_df: Clinical dataset
        protected_attr: Protected attribute name
        outcome: Outcome variable name
        use_cache: Whether to use cached results
        llm_config: Optional LLM configuration

    Returns:
        Dict with 'graph' (nx.DiGraph) and 'llm_explanation' (str)
    """
    analyzer = CausalAnalyzer(data_df, protected_attr, outcome)
    llm_enhanced = llm_config is not None if llm_config else True
    return analyzer.infer_causal_graph(use_cache=use_cache, llm_enhanced=llm_enhanced)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_path = Path(__file__).parent.parent / "data" / "sample" / "demo_data.csv"

    if data_path.exists():
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} patient records")

        result = infer_causal_graph(data, 'race', 'referral', use_cache=False)

        print("\n" + "="*60)
        print("CAUSAL ANALYSIS RESULTS")
        print("="*60)
        print(f"\nGraph nodes: {list(result['graph'].nodes())}")
        print(f"Graph edges: {list(result['graph'].edges())}")
        print("\nExplanation:")
        # Handle Unicode encoding for Windows console
        explanation = result['llm_explanation'].replace('\u2192', '->')
        print(explanation)

        analyzer = CausalAnalyzer(data, 'race', 'referral')
        analyzer.causal_graph = result['graph']
        pathways = analyzer.identify_bias_pathways()

        print("\nBias Pathways:")
        for i, path in enumerate(pathways, 1):
            print(f"{i}. {' -> '.join(path)}")
    else:
        print(f"Sample data not found at {data_path}")
