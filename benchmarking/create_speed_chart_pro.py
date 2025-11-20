import matplotlib.pyplot as plt
import numpy as np

def create_professional_speed_chart():
    """Create investor-ready speed comparison chart"""
    
    # Data from our benchmark
    systems = ['Visual Narrator', 'Claude Opus', 'GPT-4 Turbo']
    times_ms = [2.5, 5848.1, 5090.9]
    
    # Create professional figure
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Bar chart (log scale)
    colors = ['#00D4AA', '#FF6B6B', '#4E79A7']
    bars = ax1.bar(systems, times_ms, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Processing Time (ms, log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Speed Comparison: Real-time vs Batch Processing', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, time_val in zip(bars, times_ms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Chart 2: Speed multiplier
    speed_multiplier = [1, times_ms[1]/times_ms[0], times_ms[2]/times_ms[0]]
    bars2 = ax2.bar(systems, speed_multiplier, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Speed Multiplier (x faster)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Visual Narrator: {speed_multiplier[1]:.0f}x Faster', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, multiplier in zip(bars2, speed_multiplier):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{multiplier:.0f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Professional styling
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('speed_comparison_professional.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved professional speed chart: speed_comparison_professional.png")
    
    # Print key insights
    print("\n🎯 KEY INSIGHTS FROM SPEED DATA:")
    print(f"• Real-time threshold: <16ms for 60fps video")
    print(f"• Visual Narrator: {2.5}ms → Supports 400fps")
    print(f"• Claude Opus: {5848}ms → Only 0.17fps") 
    print(f"• Competitive advantage: Enables LIVE audio description")
    print(f"• Market impact: Unlocks real-time use cases impossible with batch processing")

if __name__ == "__main__":
    create_professional_speed_chart()
