from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import defaultdict
import numpy as np
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
CORS(app)

# ---------------- HOME ----------------
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


# ---------------- APRIORI ----------------
def apriori(transactions, min_support):
    total = len(transactions)

    item_count = {}
    for t in transactions:
        for item in t:
            item_count[item] = item_count.get(item, 0) + 1

    L1 = {frozenset([i]): c for i, c in item_count.items() if c / total >= min_support}
    L = [L1]
    k = 2

    while L[k-2]:
        prev = list(L[k-2].keys())
        candidates = []

        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k:
                    candidates.append(union)

        candidates = list(set(candidates))

        counts = {}
        for t in transactions:
            t = set(t)
            for c in candidates:
                if c.issubset(t):
                    counts[c] = counts.get(c, 0) + 1

        Lk = {c: cnt for c, cnt in counts.items() if cnt / total >= min_support}
        L.append(Lk)
        k += 1

    result = []
    for level in L:
        for itemset, count in level.items():
            result.append({"items": list(itemset), "count": count})

    return result


# ---------------- FP-GROWTH ----------------
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None


def build_tree(transactions, min_support):
    header = defaultdict(int)

    for t in transactions:
        for item in t:
            header[item] += 1

    header = {k: v for k, v in header.items() if v >= min_support}
    if not header:
        return None, None

    for k in header:
        header[k] = [header[k], None]

    root = FPNode(None, 1, None)

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

                if header[item][1] is None:
                    header[item][1] = new_node
                else:
                    temp = header[item][1]
                    while temp.link:
                        temp = temp.link
                    temp.link = new_node

            current = current.children[item]

    return root, header


def get_patterns(node):
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


def mine_tree(header, min_support, prefix, result):
    for item, (count, node) in sorted(header.items(), key=lambda x: x[1][0]):
        new_set = prefix.copy()
        new_set.add(item)

        result.append({"items": list(new_set), "count": count})

        patterns = get_patterns(node)

        cond_trans = []
        for path, cnt in patterns:
            for _ in range(cnt):
                cond_trans.append(path)

        cond_tree, cond_header = build_tree(cond_trans, min_support)

        if cond_header:
            mine_tree(cond_header, min_support, new_set, result)


def fp_growth(transactions, min_support):
    tree, header = build_tree(transactions, min_support)
    result = []

    if header:
        mine_tree(header, min_support, set(), result)

    return result


# ---------------- Z-SCORE ----------------
def z_score_method(data):
    try:
        data = [float(x) for x in data if str(x).strip() != ""]

        if len(data) == 0:
            return {"error": "No valid numeric data"}

        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)

        result = []
        for x in data:
            z = (x - mean) / std if std != 0 else 0
            result.append({
                "value": float(x),
                "z_score": float(z),
                "outlier": abs(z) > 3
            })

        return result
    except Exception as e:
        return {"error": str(e)}


# ---------------- ISOLATION FOREST ----------------
def isolation_forest_method(data):
    try:
        data = [float(x) for x in data if str(x).strip() != ""]

        if len(data) == 0:
            return {"error": "No valid numeric data"}

        arr = np.array(data).reshape(-1, 1)

        model = IsolationForest(contamination=0.2, random_state=42)
        model.fit(arr)
        preds = model.predict(arr)

        result = []
        for val, pred in zip(arr, preds):
            result.append({
                "value": float(val[0]),
                "outlier": True if pred == -1 else False
            })

        return result
    except Exception as e:
        return {"error": str(e)}


# ---------------- API ----------------
@app.route('/run', methods=['POST'])
def run():
    data = request.json
    algo = data['algorithm']

    if algo in ["apriori", "fp"]:
        transactions = data['transactions']
        support = data['support']

        if algo == "apriori":
            result = apriori(transactions, float(support))
        else:
            result = fp_growth(transactions, int(support))

    elif algo == "zscore":
        result = z_score_method(data['values'])

    else:
        result = isolation_forest_method(data['values'])

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)