import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis('off')

# Title
ax.text(5, 12.5, 'Agentic AI Support System Architecture', ha='center', fontsize=16, fontweight='bold', color='navy')

# Level 1: Customer (Top)

ax.add_patch(plt.Rectangle((3, 11), 4, 0.8, color='lightblue', edgecolor='black', linewidth=2))
ax.text(5, 11.4, 'CUSTOMER (User)', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 11.15, 'WhatsApp / Web Chat / Mobile App', ha='center', fontsize=9, style='italic')

# Arrow from Customer to Agent
ax.annotate('', xy=(5, 10.6), xytext=(5, 11), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Main Agent Box

ax.add_patch(plt.Rectangle((0.8, 2.5), 8.4, 8, color='lightgray', alpha=0.3, edgecolor='black', linewidth=2, linestyle='--'))
ax.text(5, 10.3, 'AGENTIC AI SUPPORT AGENT', ha='center', fontsize=13, fontweight='bold', color='darkblue')

# Box 1: Input Processing

ax.add_patch(plt.Rectangle((1.2, 8.5), 7.6, 1.2, color='lightyellow', edgecolor='black', linewidth=1.5))
ax.text(5, 9.2, '1. INPUT PROCESSING', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 8.8, 'Message Text  |  User ID / Context  |  Tracking Number', ha='center', fontsize=9)

# Arrow 1→2
ax.annotate('', xy=(5, 8.3), xytext=(5, 8.5), arrowprops=dict(arrowstyle='->', lw=1.5))

# Box 2: Intent Classifier
ax.add_patch(plt.Rectangle((1.2, 6.8), 7.6, 1.2, color='lightgreen', edgecolor='black', linewidth=1.5))
ax.text(5, 7.5, '2. INTENT CLASSIFIER', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 7.1, 'Shipping  |  Payment  |  Account  |  Complaint  |  General', ha='center', fontsize=9)

# Arrow 2→3
ax.annotate('', xy=(5, 6.6), xytext=(5, 6.8), arrowprops=dict(arrowstyle='->', lw=1.5))

# Box 3: Knowledge & APIs
ax.add_patch(plt.Rectangle((1.2, 5.1), 7.6, 1.2, color='lightcoral', edgecolor='black', linewidth=1.5))
ax.text(5, 5.8, '3. KNOWLEDGE & APIs', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 5.4, 'FAQ Database  |  Tracking API  |  Customer Data', ha='center', fontsize=9)

# Arrow 3→4
ax.annotate('', xy=(5, 4.9), xytext=(5, 5.1), arrowprops=dict(arrowstyle='->', lw=1.5))

# Box 4: Decision Engine
ax.add_patch(plt.Rectangle((1.2, 3.4), 7.6, 1.2, color='plum', edgecolor='black', linewidth=1.5))
ax.text(5, 4.1, '4. DECISION ENGINE', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 3.7, 'Can I answer automatically?  →  Auto-response', ha='center', fontsize=9)
ax.text(5, 3.5, 'Need human intervention?  →  Handoff to agent', ha='center', fontsize=9)

# Arrow 4→5 (split into two paths)
ax.annotate('', xy=(3.5, 3.2), xytext=(3.5, 3.4), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(6.5, 3.2), xytext=(6.5, 3.4), arrowprops=dict(arrowstyle='->', lw=1.5))

# Box 5a: Auto-Response (Left)
ax.add_patch(plt.Rectangle((1.2, 2), 3.2, 1.0, color='lightblue', edgecolor='black', linewidth=1.5))
ax.text(2.8, 2.45, 'AUTO-RESPONSE', ha='center', fontsize=10, fontweight='bold')
ax.text(2.8, 2.15, '(Bot replies)', ha='center', fontsize=8, style='italic')

# Box 5b: Handoff to Human (Right)
ax.add_patch(plt.Rectangle((5.6, 2), 3.2, 1.0, color='orange', edgecolor='black', linewidth=1.5))
ax.text(7.2, 2.45, 'HANDOFF TO HUMAN', ha='center', fontsize=10, fontweight='bold')
ax.text(7.2, 2.15, '(Support Agent)', ha='center', fontsize=8, style='italic')

# Arrows from both to Response Formatter
ax.annotate('', xy=(5, 1.7), xytext=(2.8, 2), arrowprops=dict(arrowstyle='->', lw=1))
ax.annotate('', xy=(5, 1.7), xytext=(7.2, 2), arrowprops=dict(arrowstyle='->', lw=1))

# Box 6: Response Formatter
ax.add_patch(plt.Rectangle((1.2, 0.8), 7.6, 0.9, color='lightsalmon', edgecolor='black', linewidth=1.5))
ax.text(5, 1.35, '5. RESPONSE FORMATTER', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 1.05, 'Text Response  |  Action Buttons  |  Links', ha='center', fontsize=9)

# Arrow to Feedback Loop
ax.annotate('', xy=(5, 0.6), xytext=(5, 0.8), arrowprops=dict(arrowstyle='->', lw=1.5))

# Box 7: Feedback Loop
ax.add_patch(plt.Rectangle((1.2, -0.3), 7.6, 0.9, color='lightcyan', edgecolor='black', linewidth=1.5))
ax.text(5, 0.25, '6. FEEDBACK LOOP', ha='center', fontsize=11, fontweight='bold')
ax.text(5, -0.05, 'User Rating  |  Sentiment Analysis  |  Model Retraining', ha='center', fontsize=9)

# Final Arrow to Customer (Response)

ax.annotate('', xy=(5, -0.7), xytext=(5, -0.3), arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
ax.text(5.2, -0.55, 'Response to Customer', ha='left', fontsize=10, fontweight='bold', color='darkgreen')


# Feedback loop arrow (curved back to top)

ax.annotate('', xy=(10.5, 9), xytext=(10.5, 0), arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='dashed', color='gray'))
ax.text(10.7, 4.5, 'Feedback for', rotation=90, ha='center', fontsize=8, color='gray')
ax.text(10.7, 3.8, 'Model Improvement', rotation=90, ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('agentic_ai_architecture.png', dpi=200, bbox_inches='tight')
plt.show()
print(" Image saved as agentic_ai_architecture.png")