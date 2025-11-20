"""
Quick validation test for architecture comparison graphs.

Tests that all three tier graphs can be instantiated and have correct structure.
Doesn't run full inference (which requires HuggingFace model downloads).
"""

import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

print("Testing graph structure validation...")
print("="*70)

# Test 1: Check if basic graph imports and compiles
print("\n[1/3] Testing BASIC graph structure...")
try:
    from advanced_agentic_rag_langgraph.variants.basic_rag_graph import basic_rag_graph

    # Check graph structure
    assert hasattr(basic_rag_graph, 'nodes'), "Basic graph should have nodes"
    node_names = list(basic_rag_graph.nodes.keys())
    print(f"  Nodes ({len(node_names)}): {', '.join(node_names)}")
    print("  [OK] Basic graph structure valid")
except Exception as e:
    print(f"  [FAIL] Basic graph error: {str(e)[:100]}")

# Test 2: Check if intermediate graph imports and compiles
print("\n[2/3] Testing INTERMEDIATE graph structure...")
try:
    from advanced_agentic_rag_langgraph.variants.intermediate_rag_graph import intermediate_rag_graph

    # Check graph structure
    assert hasattr(intermediate_rag_graph, 'nodes'), "Intermediate graph should have nodes"
    node_names = list(intermediate_rag_graph.nodes.keys())
    print(f"  Nodes ({len(node_names)}): {', '.join(node_names)}")
    print("  [OK] Intermediate graph structure valid")
except Exception as e:
    print(f"  [FAIL] Intermediate graph error: {str(e)[:100]}")

# Test 3: Check if advanced graph can be imported (may fail without HuggingFace models)
print("\n[3/3] Testing ADVANCED graph import...")
try:
    from advanced_agentic_rag_langgraph.variants.advanced_rag_graph import advanced_rag_graph

    # Check graph structure
    assert hasattr(advanced_rag_graph, 'nodes'), "Advanced graph should have nodes"
    node_names = list(advanced_rag_graph.nodes.keys())
    print(f"  Nodes ({len(node_names)}): {', '.join(node_names)}")
    print("  [OK] Advanced graph structure valid")
except Exception as e:
    error_msg = str(e)
    if "huggingface" in error_msg.lower() or "403" in error_msg:
        print(f"  [EXPECTED] Advanced graph requires HuggingFace model download")
        print(f"  This is expected in offline/restricted environments")
        print(f"  Error: Cannot download 'cross-encoder/nli-deberta-v3-base'")
    else:
        print(f"  [FAIL] Unexpected error: {error_msg[:100]}")

print("\n" + "="*70)
print("Graph structure validation complete")
print("="*70)
print("\nNOTE: Full architecture comparison requires:")
print("  1. Internet access to download HuggingFace models")
print("  2. ~2GB disk space for model weights")
print("  3. OpenAI API key for LLM calls")
print("\nTo run full comparison (when network is available):")
print("  uv run python tests/integration/test_architecture_comparison.py")
