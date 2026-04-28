from itertools import combinations

def apriori(transactions, min_support):
    total = len(transactions)

    # Step 1: Generate L1
    item_count = {}
    for t in transactions:
        for item in t:
            item_count[item] = item_count.get(item, 0) + 1

    L1 = {frozenset([i]): c for i, c in item_count.items() if c / total >= min_support}
    L = [L1]

    k = 2

    while L[k-2]:
        prev_itemsets = list(L[k-2].keys())
        candidates = []

        # JOIN step
        for i in range(len(prev_itemsets)):
            for j in range(i+1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    candidates.append(union)

        # Remove duplicates
        candidates = list(set(candidates))

        # COUNT support
        counts = {}
        for t in transactions:
            t_set = set(t)
            for c in candidates:
                if c.issubset(t_set):
                    counts[c] = counts.get(c, 0) + 1

        # FILTER by min_support
        Lk = {c: cnt for c, cnt in counts.items() if cnt / total >= min_support}

        L.append(Lk)
        k += 1

    return L


# 🔸 Example Usage
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter']
]

min_support = 0.4

result = apriori(transactions, min_support)

print("Frequent Itemsets (Apriori):")
for level in result:
    for itemset, count in level.items():
        print(set(itemset), ":", count)