import matplotlib.pyplot as plt
import numpy as np

def clean_price(price_val):
    """Convert price string/float to float"""
    if isinstance(price_val, (float, int)):
        return float(price_val)
    
    # String cleanup
    s = str(price_val).strip()
    s = s.replace('£', '').replace('$', '').replace('€', '').replace(',', '')
    try:
        return float(s)
    except:
        return 0.0

def plot_receipt_dashboard(data):
    """
    Data format: list of lists
    Row 0: [Store Name]
    Row 1..N-1: [Item, Category, Price]
    Row N: [Total, Note, Sum]
    """
    if not data or len(data) < 3:
        print("Not enough data to visualize")
        return

    # 1. Parse Data
    store_name = data[0][0]
    
    # Separate items from footer
    # Heuristic: Check last row for "total"
    items_rows = data[1:]
    total_row = None
    
    if items_rows and str(items_rows[-1][0]).lower() == "total":
        total_row = items_rows[-1]
        items_rows = items_rows[:-1]
        
    categories = {}
    item_names = []
    item_prices = []
    
    for row in items_rows:
        if len(row) < 3: continue
        name = row[0]
        cat = row[1].lower().capitalize() if row[1] else "Uncategorized"
        price = clean_price(row[2])
        
        item_names.append(name)
        item_prices.append(price)
        
        categories[cat] = categories.get(cat, 0) + price

    # 2. Setup Dashboard
    plt.style.use('bmh') # Clean aesthetic style
    fig = plt.figure(figsize=(14, 8))
    grid = plt.GridSpec(2, 2, figure=fig)
    
    # Title
    total_str = f"Total: {total_row[2]}" if total_row else ""
    fig.suptitle(f"Receipt Analysis: {store_name}  |  {total_str}", fontsize=16, fontweight='bold')

    # 3. Pie Chart (Categories)
    ax1 = fig.add_subplot(grid[0, 0])
    
    cats = list(categories.keys())
    vals = list(categories.values())
    
    if vals:
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(cats)))
        wedges, texts, autotexts = ax1.pie(vals, labels=cats, autopct='%1.1f%%', 
                                         startangle=90, colors=colors, pctdistance=0.85)
        
        # Donut style
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        ax1.add_artist(centre_circle)
        
        ax1.set_title("Spending by Category", fontsize=12)
        plt.setp(autotexts, size=9, weight="bold")

    # 4. Bar Chart (Top Items)
    ax2 = fig.add_subplot(grid[0, 1])
    
    # Sort by price
    zipped = sorted(zip(item_names, item_prices), key=lambda x: x[1])
    idx = range(len(zipped))
    sorted_names = [x[0] for x in zipped]
    sorted_prices = [x[1] for x in zipped]
    
    if sorted_prices:
        bars = ax2.barh(idx, sorted_prices, color='cornflowerblue')
        ax2.set_yticks(idx)
        ax2.set_yticklabels(sorted_names, fontsize=9)
        ax2.set_xlabel("Price")
        ax2.set_title("Itemized Cost Breakdown")
        
        # Add value labels
        for i, v in enumerate(sorted_prices):
            ax2.text(v, i, f" {v:.2f}", va='center', fontsize=8, color='black')

    # 5. Summary Text
    ax3 = fig.add_subplot(grid[1, :])
    ax3.axis('off')
    
    # Construct summary stats
    valid_items = len(item_prices)
    avg_price = sum(item_prices) / valid_items if valid_items else 0
    most_expensive = sorted_names[-1] if sorted_names else "N/A"
    
    summary_text = (
        f"Quick Stats:\n"
        f"-----------------\n"
        f"• Number of Items: {valid_items}\n"
        f"• Average Item Cost: {avg_price:.2f}\n"
        f"• Most Expensive: {most_expensive}\n"
        f"• Top Category: {max(categories, key=categories.get) if categories else 'N/A'}"
    )
    
    ax3.text(0.5, 0.5, summary_text, ha='center', va='center', 
             fontsize=12, family='monospace', 
             bbox=dict(boxstyle="round,pad=1", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
