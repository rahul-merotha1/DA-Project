from collections import defaultdict

class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None


def build_tree(transactions, min_support):
    header = defaultdict(int)

    # First scan: count frequency
    for t in transactions:
        for item in t:
            header[item] += 1

    # Remove infrequent items
    header = {k: v for k, v in header.items() if v >= min_support}
    if not header:
        return None, None

    # Header table structure
    for k in header:
        header[k] = [header[k], None]

    root = FPNode(None, 1, None)

    # Second scan: build tree
    for t in transactions:
        items = [i for i in t if i in header]
        items.sort(key=lambda x: header[x][0], reverse=True)

        current = root
        for item in items:
            if item in current.children:
                current.children[item].count += 1
            else:
                new_node = FPNode(item, 1, current)
                current.children[item] = new_node

                # Update header link
                if header[item][1] is None:
                    header[item][1] = new_node
                else:
                    temp = header[item][1]
                    while temp.link:
                        temp = temp.link
                    temp.link = new_node

            current = current.children[item]

    return root, header


def get_conditional_pattern_base(node):
    patterns = []
    while node:
        path = []
        parent = node.parent

        while parent and parent.item:
            path.append(parent.item)
            parent = parent.parent

        if path:
            patterns.append((path, node.count))

        node = node.link

    return patterns


def mine_tree(header, min_support, prefix, freq_items):
    for item, (count, node) in sorted(header.items(), key=lambda x: x[1][0]):
        new_set = prefix.copy()
        new_set.add(item)

        freq_items.append((new_set, count))

        # Get conditional pattern base
        patterns = get_conditional_pattern_base(node)

        cond_transactions = []
        for path, cnt in patterns:
            for _ in range(cnt):
                cond_transactions.append(path)

        # Build conditional tree
        cond_tree, cond_header = build_tree(cond_transactions, min_support)

        if cond_header:
            mine_tree(cond_header, min_support, new_set, freq_items)


# 🔸 Example Usage
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter']
]

min_support = 2   # NOTE: count (not percentage)

tree, header = build_tree(transactions, min_support)

freq_items = []

if header:
    mine_tree(header, min_support, set(), freq_items)

print("\nFrequent Itemsets (FP-Growth):")
for itemset, count in freq_items:
    print(itemset, ":", count)