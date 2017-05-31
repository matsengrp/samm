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
            MotifTreeNode("%s->n (%d)" % (motif, mutating_pos), not is_zero(theta_val))
        )
        if self.is_per_target:
            for target_nuc_idx, target_nuc in enumerate(NUCLEOTIDES):
                if target_nuc != mutating_nuc:
                    node_name = "%s->%s (%d)" % (motif, target_nuc, mutating_pos)
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
                    # if not the same rule and the rule we are considering could be a subset of the current rule
                    if rule_idx != i and rule_motif_len < this_motif_len:
                        if mutating_pos - rule_left_flank_len >= 0:
                            start_idx = mutating_pos - rule_left_flank_len
                            submotif = motif[start_idx : start_idx + rule_motif_len]
                            if len(submotif) == rule_motif_len:
                                # We have a match!
                                match_theta_idx = self.feat_generator.motif_dict[submotif][rule_left_flank_len]
                                nodes[raw_theta_idx][0].add_parent(nodes[match_theta_idx][0])
                                nodes[match_theta_idx][0].add_child(nodes[raw_theta_idx][0])
                                if self.is_per_target:
                                    for per_target_node_idx in range(1, NUM_NUCLEOTIDES):
                                        nodes[raw_theta_idx][per_target_node_idx].add_parent(nodes[match_theta_idx][per_target_node_idx])
                                        nodes[match_theta_idx][per_target_node_idx].add_child(nodes[raw_theta_idx][per_target_node_idx])
        return nodes

    def _push_up_free_status(self, node):
        """
        Now that all the nodes should be updated with the free status, just check if
        children are all nonzero. If so, remove the parent node.
        """
        if not node.is_leaf() and not node.is_free:
            children_status = [child.is_free for child in node.children]
            if np.all(children_status) != node.is_free:
                node.is_free = np.all(children_status)
                return True
        return False

    def _prune_subtree_nonzero(self, node):
        """
        Now that all the nodes should be updated with the free status, just check if
        children are all nonzero. If so, remove the parent node.
        """
        if not node.is_leaf():
            children_status = [child.is_free for child in node.children]
            node.should_remove = np.all(children_status)
        else:
            node.should_remove = not node.is_free

    def _create_theta_mask(self):
        removal_mask = np.zeros(self.theta.shape, dtype=bool)
        for node_idx, row_nodes in enumerate(self.all_tree_nodes):
            for col_idx, node in enumerate(row_nodes):
                removal_mask[node_idx, col_idx] = not node.should_remove and node.is_free
        return removal_mask

    def _push_down_free_status(self, node):
        """
        Determine which child nodes are completely free of the parent node. To do this,
        clean up the free status in the tree. Push down the free status from parents to children
        if at most one child is currently constrained to be zero. If the parent node is constrained
        at zero but the children are all nonzero, update the parent node to be free.
        """
        if not node.is_leaf() and node.is_free:
            children_status = np.array([child.is_free for child in node.children])
            if np.sum(children_status) == len(node.children) - 1:
                free_child_idx = np.where(~children_status)[0][0]
                node.children[free_child_idx].is_free = True
                return True
        return False

    def prune(self):
        # find all free nodes
        has_change = True
        while has_change:
            # print "haschange"
            has_change = False
            for row_nodes in self.all_tree_nodes:
                for node in row_nodes:
                    has_change |= self._push_down_free_status(node)

            for row_nodes in self.all_tree_nodes:
                for node in row_nodes:
                    has_change |= self._push_up_free_status(node)

        # find all redundant parents
        for row_nodes in self.all_tree_nodes:
            for node in row_nodes:
                self._prune_subtree_nonzero(node)

        # for row_nodes in self.all_tree_nodes:
        #     for node in row_nodes:
        #         if not node.is_leaf() and node.is_free and not node.should_remove:
        #             print str(node)
        #             for c in node.children:
        #                 if not c.is_nonzero:
        #                     print "    ", str(c)

        # for row_nodes in self.all_tree_nodes:
        #     for node in row_nodes:
        #         if not node.is_leaf():
        #             for cleanup_ns in self.all_tree_nodes:
        #                 for cleanup_n in cleanup_ns:
        #                     cleanup_n.final_explore = False
        #
        #             self._find_cycles(node, [], "")


        keep_mask = self._create_theta_mask()
        return keep_mask

    # def _find_cycles(self, node, prev_child_nodes, path_str):
    #     if not node.final_explore:
    #         node.final_explore = True
    #     else:
    #         print "path_str", path_str, str(node)
    #         print "OMG HUH"
    #         1/0
    #     if node.is_free:
    #         if np.sum([not c.is_free for c in node.children]) == 2:
    #             for c in node.children:
    #                 if not c.is_free and c not in prev_child_nodes:
    #                     for p in c.parents:
    #                         if str(p) != str(node):
    #                             self._find_cycles(p, prev_child_nodes + [c], "%s, %s, %s" % (path_str, str(node), str(c)))


class MotifTreeNode:
    def __init__(self, name, is_nonzero):
        self.name = name
        self.is_nonzero = is_nonzero
        self.should_remove = False
        self.is_free = is_nonzero
        self.explored_state = 0
        self.children = []
        self.parents = []

    def is_leaf(self):
        return len(self.children) == 0

    def is_tree_head(self):
        return len(self.parents) == 0

    def set_removal_status(self, status):
        self.should_remove = status

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parents.append(parent)

    def __str__(self):
        return self.name
