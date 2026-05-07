"""
Visual Narrator 3B - Technical Comparison Demo
Updated with accurate, verified claims (January 2026)

Hugging Face Space: https://huggingface.co/spaces/Ytgetahun/visual-narrator-comparison
"""

import gradio as gr

# =============================================================================
# VERIFIED BENCHMARK DATA (Source: Accurate Benchmark Claims Jan 2026)
# =============================================================================

SPEED_DATA = {
    "Visual Narrator 3B": {
        "latency_ms": 2.4,
        "real_time": True,
        "fps_capable": 400,
    },
    "GPT-4 Turbo": {
        "latency_ms": 2344,
        "real_time": False,
        "fps_capable": 0.43,
    },
    "Claude Opus": {
        "latency_ms": 3536,
        "real_time": False,
        "fps_capable": 0.28,
    },
}

QUALITY_DATA = {
    "Visual Narrator 3B": {"adj_density": 2.0, "semantic_accuracy": 71.6},
    "Claude Sonnet 4.5": {"adj_density": 2.0, "semantic_accuracy": 64.2},
    "GPT-4 Turbo": {"adj_density": 2.0, "semantic_accuracy": 66.8},
}

COST_DATA = {
    "Visual Narrator": {"per_frame": 0.00, "monthly_1m": 900},
    "GPT-4 Vision": {"per_frame": 0.083, "monthly_1m": 83000},
    "Claude Vision": {"per_frame": 0.252, "monthly_1m": 252000},
}

# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_speed_comparison():
    """Generate speed benchmark comparison."""
    vn = SPEED_DATA["Visual Narrator 3B"]
    gpt4 = SPEED_DATA["GPT-4 Turbo"]
    claude = SPEED_DATA["Claude Opus"]

    speed_vs_gpt4 = gpt4["latency_ms"] / vn["latency_ms"]
    speed_vs_claude = claude["latency_ms"] / vn["latency_ms"]

    return f"""
## Video-to-Text Speed Benchmark

**What's measured:** End-to-end latency from video frame input to narration output.

| Model | Latency | Speed vs VN | Real-Time? |
|-------|---------|-------------|------------|
| **Visual Narrator 3B** | **{vn['latency_ms']}ms** | — | Yes ({vn['fps_capable']}+ FPS) |
| GPT-4 Turbo | {gpt4['latency_ms']:,}ms | {speed_vs_gpt4:.0f}x slower | No |
| Claude Opus | {claude['latency_ms']:,}ms | {speed_vs_claude:.0f}x slower | No |

### What This Proves
Visual Narrator can narrate **live video in real-time**.
Competitor APIs are limited to batch/offline processing due to 2-3 second latency.

### Methodology Note
- Visual Narrator: Local/edge GPU inference
- Competitors: Cloud API round-trip (includes network latency)
- This reflects real-world deployment conditions
"""

def create_quality_comparison():
    """Generate quality benchmark comparison."""
    vn = QUALITY_DATA["Visual Narrator 3B"]
    claude = QUALITY_DATA["Claude Sonnet 4.5"]
    gpt4 = QUALITY_DATA["GPT-4 Turbo"]

    return f"""
## Text-to-Text Quality Benchmark

**What's measured:** Descriptive language richness (adjectives per description).

| Model | Adj/Description | Semantic Accuracy |
|-------|-----------------|-------------------|
| **Visual Narrator 3B** | **{vn['adj_density']}** | **{vn['semantic_accuracy']}%** |
| Claude Sonnet 4.5 | {claude['adj_density']} | {claude['semantic_accuracy']}% |
| GPT-4 Turbo | {gpt4['adj_density']} | {gpt4['semantic_accuracy']}% |

### What This Proves
Our language generation quality **matches Claude-tier output**.
We don't sacrifice quality for speed.

### Historical Context
Early training achieved 3.62 adj/desc (+81% vs Claude).
We **intentionally reduced to 2.0** after determining higher density = "fluff".
Claude's 2.0 was the **correct quality target**, not something to exceed.
"""

def create_cost_comparison():
    """Generate cost comparison."""
    vn = COST_DATA["Visual Narrator"]
    gpt4 = COST_DATA["GPT-4 Vision"]
    claude = COST_DATA["Claude Vision"]

    savings_vs_gpt4 = gpt4["monthly_1m"] / vn["monthly_1m"]
    savings_vs_claude = claude["monthly_1m"] / vn["monthly_1m"]

    return f"""
## Economics at Scale

**Scenario:** Processing 1 million videos per month

| Provider | Cost/Frame | Monthly Cost (1M videos) |
|----------|------------|--------------------------|
| **Visual Narrator** | **${vn['per_frame']:.2f}** | **${vn['monthly_1m']:,}** (fixed) |
| GPT-4 Vision | ${gpt4['per_frame']:.3f} | ${gpt4['monthly_1m']:,} |
| Claude Vision | ${claude['per_frame']:.3f} | ${claude['monthly_1m']:,} |

### Cost Advantage
- **{savings_vs_gpt4:.0f}x cheaper** than GPT-4 Vision
- **{savings_vs_claude:.0f}x cheaper** than Claude Vision
- **Zero marginal cost** per additional frame (fixed infrastructure)
"""

def create_summary():
    """Generate executive summary."""
    return """
## The Honest Pitch

### What We Don't Claim
- We don't claim to "beat" Claude on language quality
- We don't claim "trillion-parameter" comparisons
- We don't claim adjective density superiority

### What We Do Claim (Verified)
- We **MATCH** premium API quality (2.0 adj/desc)
- We process video **976x faster** (2.4ms vs 2,344ms)
- We enable **real-time markets** competitors cannot serve
- We cost **90-280x less** at scale

### The Unlock
> Real-time video narration at premium quality—
> a combination no API-based competitor can match.

This enables: live broadcasting, streaming accessibility,
real-time content creation—markets that API latency blocks.
"""

def create_sample_output():
    """Show sample model output."""
    return """
## Sample Output

**Input:** Video frame of urban night scene

**Visual Narrator 3B:**
> "A sleek automobile navigates the urban landscape at night,
> neon lights reflecting off wet pavement as pedestrians
> move through crosswalks beneath glowing storefronts."

**Characteristics:**
- Professional narrative flow
- Appropriate descriptive density (not over-decorated)
- Spatial and temporal awareness
- Suitable for audio description / accessibility
"""

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(
    title="Visual Narrator 3B - Technical Comparison",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # Visual Narrator 3B - Technical Comparison

    **Matching Premium Quality at Real-Time Speed**

    A specialized 3B parameter model that matches Claude-quality descriptions
    while enabling real-time video narration that API-based models cannot achieve.

    ---

    ### Understanding the Two Benchmark Types

    | Benchmark | What's Measured | Our Advantage |
    |-----------|-----------------|---------------|
    | **Video-to-Text (Speed)** | Frame processing latency | 976x faster |
    | **Text-to-Text (Quality)** | Descriptive language richness | Parity with Claude |

    These measure **different capabilities** and should not be conflated.

    ---
    """)

    with gr.Tabs():
        with gr.Tab("Speed Benchmark"):
            gr.Markdown(create_speed_comparison())

        with gr.Tab("Quality Benchmark"):
            gr.Markdown(create_quality_comparison())

        with gr.Tab("Cost Analysis"):
            gr.Markdown(create_cost_comparison())

        with gr.Tab("Sample Output"):
            gr.Markdown(create_sample_output())

        with gr.Tab("Summary"):
            gr.Markdown(create_summary())

    gr.Markdown("""
    ---

    ### Links
    - [Model Repository](https://huggingface.co/Ytgetahun/visual-narrator-llm)
    - [Full Documentation](https://github.com/yonnastgetahun/visual-narrator-docs)

    ### Methodology
    All claims verified and documented in "Visual Narrator 3B - Accurate Benchmark Claims (Jan 2026)"

    ---
    *Last updated: January 2026*
    """)

# =============================================================================
# LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo.launch()
