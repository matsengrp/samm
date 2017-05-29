from common import *

class ModelPruner:
    def __init__(self, feat_generator, full_feat_generator, theta):
        self.feat_generator = feat_generator
        self.full_feat_generator = full_feat_generator
        self.agg_motif_list = self.full_feat_generator.motif_list
        self.motif_list = self.feat_generator.motif_list
        self.mutating_pos_list = self.feat_generator.mutating_pos_list
        self.theta = theta
        self.is_per_target = self.theta.shape[1] == NUM_NUCLEOTIDES + 1

        self.all_tree_nodes = self._construct_motif_tree()

    def _create_nodes(self, motif_idx, motif, mutating_pos):
        mutating_nuc = motif[mutating_pos]
        nodes = []
        theta_val = self.theta[motif_idx, 0]
        nodes.append(
            MotifTreeNode("%s->n" % (motif), not is_zero(theta_val))
        )
        if self.is_per_target:
            for target_nuc_idx, target_nuc in enumerate(NUCLEOTIDES):
                if target_nuc != mutating_nuc:
                    node_name = "%s->%s" % (motif, target_nuc)
                    theta_val = self.theta[motif_idx, target_nuc_idx + 1]
                    nodes.append(
                        MotifTreeNode(node_name, not is_zero(theta_val))
                    )
        return nodes

    def _construct_motif_tree(self):
        # Create nodes, but nothing is connected at this point!
        nodes = []
        for motif_idx, (motif, mutating_pos) in enumerate(zip(self.motif_list, self.mutating_pos_list)):
            new_nodes = self._create_nodes(motif_idx, motif, mutating_pos)
            nodes.append(new_nodes)

        # Figure out the submotif feature generation rules
        feat_gen_rules = [(feat_gen.motif_len, feat_gen.left_motif_flank_len) for feat_gen in self.feat_generator.feat_gens]

        # Figure out the hierarchy
        # For each raw motif, manually calculate any submotifs, then append as parents
        for i, feat_gen in enumerate(self.feat_generator.feat_gens):
            this_feat_gen_rule = feat_gen_rules[i]
            this_motif_len = this_feat_gen_rule[0]
            # this_left_flank_len = this_feat_gen_rule[1]
            mutating_pos = feat_gen.left_motif_flank_len
            for m_idx, motif in enumerate(feat_gen.motif_list):
                raw_theta_idx = self.feat_generator.feat_offsets[i] + m_idx

                if self.is_per_target:
                    for per_target_node_idx in range(1, NUM_NUCLEOTIDES):
                        nodes[raw_theta_idx][per_target_node_idx].add_parent(nodes[raw_theta_idx][0])
                        nodes[raw_theta_idx][0].add_child(nodes[raw_theta_idx][per_target_node_idx])

                for rule_idx, feat_gen_rule in enumerate(feat_gen_rules):
                    rule_motif_len = feat_gen_rule[0]
                    rule_left_flank_len = feat_gen_rule[1]
                    if rule_idx != i and rule_motif_len < this_motif_len:
                        if mutating_pos - rule_left_flank_len >= 0:
                            start_idx = mutating_pos - rule_left_flank_len
                            submotif = motif[start_idx : start_idx + rule_motif_len]
                            if len(submotif) == rule_motif_len:
                                # We have a match!
                                submotif_match_idx = self.feat_generator.feat_gens[rule_idx].motif_dict[submotif]
                                match_theta_idx = self.feat_generator.feat_offsets[rule_idx] + submotif_match_idx
                                nodes[raw_theta_idx][0].add_parent(nodes[match_theta_idx][0])
                                nodes[match_theta_idx][0].add_child(nodes[raw_theta_idx][0])
                                if self.is_per_target:
                                    for per_target_node_idx in range(1, NUM_NUCLEOTIDES):
                                        nodes[raw_theta_idx][per_target_node_idx].add_parent(nodes[match_theta_idx][per_target_node_idx])
                                        nodes[match_theta_idx][per_target_node_idx].add_child(nodes[raw_theta_idx][per_target_node_idx])
        return nodes

    def _prune_subtree_nonzero(self, node):
        if node.is_leaf():
            return node.is_nonzero

        if node.explored:
            return node.should_remove

        children_status = [self._prune_subtree_nonzero(child) for child in node.children]
        node.explored = True
        if np.all(children_status):
            node.set_removal_status(True)
            return True
        else:
            return False

    def _create_theta_mask(self):
        removal_mask = np.zeros(self.theta.shape, dtype=bool)
        for node_idx, row_nodes in enumerate(self.all_tree_nodes):
            for col_idx, node in enumerate(row_nodes):
                removal_mask[node_idx, col_idx] = node.should_remove
        return removal_mask

    def prune(self):
        for row_nodes in self.all_tree_nodes:
            for node in row_nodes:
                self._prune_subtree_nonzero(node)

        removal_mask = self._create_theta_mask()
        return removal_mask

class MotifTreeNode:
    def __init__(self, name, is_nonzero):
        self.name = name
        self.is_nonzero = is_nonzero
        self.should_remove = False
        self.explored = False
        self.children = []
        self.parents = []

    def is_leaf(self):
        return len(self.children) == 0

    def set_removal_status(self, status):
        self.should_remove = status

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parents.append(parent)

    def __str__(self):
        return self.name
