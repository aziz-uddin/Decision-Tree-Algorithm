#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace std;


class Dataset {
public:
    vector<vector<int>> X; // feature matrix (10 binary features)
    vector<int> y;         // class labels

    // load data from file
    bool load(string filename) {
        ifstream in(filename);

        if (!in) {
            cout << "Error opening file\n";
            return false;
        }

        while (true) {
            vector<int> row(10);
            int label;

            // read 10 features
            for (int i = 0; i < 10; i++) {
                if (!(in >> row[i]))
                    return true; // stop at end of file
            }

            // read class label
            if (!(in >> label))
                return true;

            X.push_back(row);
            y.push_back(label);
        }
    }
};

class Node {
public:
    bool isLeaf;          // true if leaf node
    int feature;          // feature used for split
    int prediction;       // class prediction (for leaf)

    Node* left;           // left child
    Node* right;          // right child

    vector<int> indices;  // indices of data points in this node

    Node() {
        isLeaf = true;
        feature = -1;
        prediction = 0;
        left = nullptr;
        right = nullptr;
    }
};

class DecisionTree {
private:
    Dataset* data;
    Node* root;

    // find most common class in a node
    int majorityClass(const vector<int>& idx) {
        vector<int> count(3, 0);

        for (int i : idx)
            count[data->y[i]]++;

        return max_element(count.begin(), count.end()) - count.begin();
    }

    // calculate Gini impurity
    double gini(const vector<int>& idx) {
        if (idx.empty()) return 0.0;

        vector<int> count(3, 0);

        for (int i : idx)
            count[data->y[i]]++;

        double n = idx.size();
        double score = 1.0;

        for (int c : count) {
            double p = c / n;
            score -= p * p;
        }

        return score;
    }

    // helper class to store best split
    class Split {
    public:
        double gain;
        int feature;
        vector<int> L, R;

        Split() {
            gain = 0;
            feature = -1;
        }
    };

    // find best feature to split on
    bool bestSplit(const vector<int>& idx, Split& best) {

        double parent = gini(idx);
        best = Split();

        // try all 10 features
        for (int f = 0; f < 10; f++) {

            vector<int> left, right;

            // split data based on feature value
            for (int i : idx) {
                if (data->X[i][f] == 0)
                    left.push_back(i);
                else
                    right.push_back(i);
            }

            // ignore useless splits
            if (left.empty() || right.empty())
                continue;

            // compute weighted Gini after split
            double child =
                (left.size() * gini(left) +
                 right.size() * gini(right))
                / idx.size();

            // gain is the improvement in purity
            double gain = (parent - child) * idx.size();

            // keep best split
            if (gain > best.gain) {
                best.gain = gain;
                best.feature = f;
                best.L = left;
                best.R = right;
            }
        }

        return best.feature != -1;
    }

    // priority queue item
    class Item {
    public:
        double gain;
        Node* node;

        Item(double g, Node* n) {
            gain = g;
            node = n;
        }

        // max heap (highest gain first)
        bool operator<(const Item& other) const {
            return gain < other.gain;
        }
    };

    // recursive prediction
    int predictRow(Node* node, const vector<int>& row) {
        if (node->isLeaf)
            return node->prediction;

        if (row[node->feature] == 0)
            return predictRow(node->left, row);
        else
            return predictRow(node->right, row);
    }

    // delete tree to avoid memory leaks
    void destroy(Node* node) {
        if (!node) return;

        destroy(node->left);
        destroy(node->right);
        delete node;
    }

public:
    DecisionTree() {
        root = nullptr;
    }


    void train(Dataset& d, int maxRules) {

        data = &d;

        // start with root node containing all data
        root = new Node();

        root->indices.resize(d.X.size());
        iota(root->indices.begin(), root->indices.end(), 0);

        root->prediction = majorityClass(root->indices);

        priority_queue<Item> pq;

        // compute first split
        Split first;
        if (bestSplit(root->indices, first))
            pq.push(Item(first.gain, root));

        int rules = 0;

        // grow tree
        while (!pq.empty() && rules < maxRules) {

            Item top = pq.top();
            pq.pop();

            Node* leaf = top.node;

            // skip if already split
            if (!leaf->isLeaf)
                continue;

            Split sp;
            if (!bestSplit(leaf->indices, sp))
                continue;

            // convert leaf into rule node
            leaf->isLeaf = false;
            leaf->feature = sp.feature;

            // create children
            leaf->left = new Node();
            leaf->right = new Node();

            leaf->left->indices = sp.L;
            leaf->right->indices = sp.R;

            // assign predictions
            leaf->left->prediction = majorityClass(sp.L);
            leaf->right->prediction = majorityClass(sp.R);

            // push children into queue
            Split a, b;

            if (bestSplit(sp.L, a))
                pq.push(Item(a.gain, leaf->left));

            if (bestSplit(sp.R, b))
                pq.push(Item(b.gain, leaf->right));

            rules++;
        }
    }

    double accuracy(Dataset& test) {

        int correct = 0;

        for (size_t i = 0; i < test.X.size(); i++) {
            if (predictRow(root, test.X[i]) == test.y[i])
                correct++;
        }

        return 100.0 * correct / test.X.size();
    }

    ~DecisionTree() {
        destroy(root);
    }
};

int main() {

    Dataset train, test;

    train.load("training.dat");
    test.load("testing.dat");

    cout << "Priority Queue Decision Tree\n\n";

    // test different number of rules
    for (int rules = 1; rules <= 20; rules += 1) {

        DecisionTree tree;
        tree.train(train, rules);

        cout << "Rules "
             << rules
             << " Accuracy: "
             << tree.accuracy(test)
             << "%\n";
    }

    return 0;
}