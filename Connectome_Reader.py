################################################################################
# Copyright Bardozzo et al.
# Graph Net Spektral extention + multi-dyadic algorithms
################################################################################

from igraph import *
from itertools import combinations
import numpy as np
import scipy.sparse as sp
import random
import itertools
from itertools import chain


class Connectome_Reader():
    def __init__(self, name_file, **kwargs):
        self.name_file = name_file
        self.graph_el = 0

    def read(self, sym_flag=1, pp=0.1, seed=1):
        self.graph_el = Graph.Read_GraphML(self.name_file)

        #summary(self.graph_el)
        #print(str(np.max(self.graph_el.vs()["soma_pos"])))
        #input()
        #for el in self.graph_el.es():
        #    if el["synapse_type"] == "E":
        #        source_vertex_id = el.source
        #        target_vertex_id = el.target
        #        print(self.graph_el.vs[source_vertex_id])
        #        print(self.graph_el.vs[target_vertex_id])

        #print(self.graph_el.vs()["cell_class"])

        #input()
        if sym_flag  == 2:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=False, multiple=False)
        elif sym_flag == 3:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=False, multiple=True)
        elif sym_flag == 4:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=True, multiple=False)
        elif sym_flag == 5:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=True, multiple=True)

        #print(summary(self.graph_el ))
        return self.graph_el

    def make_it_undirected(self):
        self.graph_el = self.graph_el.as_undirected()

    def get_it_undirected(self):
        return  self.graph_el.as_undirected()

    def get_edge_list(self):
        l1 = []
        l2 = []
        for el in self.graph_el.get_edgelist():
            l1.append(el[0])
            l2.append(el[1])

        return l1, l2

    def get_paths(self, length_effect=3):
        l1 = []
        l1_func = []
        l2 = []
        l2_func = []
        l3 = []

        for el in [p for p in itertools.product(np.asarray(self.graph_el.vs().indices), repeat=2)]:
            ll = self.graph_el.get_all_shortest_paths(el[0], el[1], mode=None)
            if ll != []:
                if len(ll[0]) == length_effect :
                    for el2  in ll:
                        if len(el2) ==length_effect:
                            l3.append(el2)
                            l1.append(el[0])
                            l2.append(el[1])
                            el_f1 = self.graph_el.vs()[el[0]]["role"]
                            if (el_f1 == "NA"):
                                el_f1 = "I"
                            l1_func.append(el_f1)
                            el_f2 = self.graph_el.vs()[el[1]]["role"]
                            if (el_f2 == "NA"):
                                el_f2 = "I"
                            l2_func.append(el_f2)

        return l1, l2, l3, l1_func, l2_func

    def get_triplet_list(self):
        l1 = []
        l2 = []
        l3 = []
        for el in [p for p in itertools.product(np.asarray(self.graph_el.vs().indices), repeat=2)]:
            ll = self.graph_el.get_all_shortest_paths(el[0], el[1], mode=None)
            if ll != []:
                if len(ll[0]) == 3 :
                    for el2  in ll:
                        if len(el2) ==3:

                            l3.append(el2)
                            l1.append(el[0])
                            l2.append(el[1])

        # generate dict edges
        l0 = []
        for i in range(0, len(l2),1):
            l0.append([l1[i], l2[i]])

        ll_0 = list([set(x) for x in l0])
        edges_dict = {index: list(value) for index, value in enumerate(ll_0)}

        # generate dict triplets
        ll = list([set(x) for x in l3])
        target_dict = {index: list(value) for index, value in enumerate(ll)}

        # print("--------")
        # print(l1[14600])
        # print(l2[14600])
        # print(edges_dict.get(14600))   #[a, b]
        # print(target_dict.get(14600))  #[
        # print("--------")

        # input()

        return edges_dict, target_dict, l1, l2

    def get_role_list(self):
        edge_list1, edge_list2 = self.get_edge_list()
        role_list_l1 = []
        role_list_l2 = []

        for el in edge_list1:
            el1 = self.graph_el.vs()[el]["role"]
            if el1 == "NA":
                el1 = "I"
            role_list_l1.append(el1)

        for el in edge_list2:
            el2 = self.graph_el.vs()[el]["role"]
            if el2 == "NA":
                el2 = "I"
            role_list_l2.append(el2)

        return role_list_l1, role_list_l2

    def get_syn_list(self):
        edge_list = self.graph_el.get_edgelist()
        syn_list = []

        for el in edge_list:
            gg_el = self.graph_el.es[self.graph_el.get_eid(el[0], el[1])]
            el1 = gg_el["synapse_type"]
            syn_list.append(el1)

        return syn_list

    def get_weight_edge(self):
        edge_list = self.graph_el.get_edgelist()
        syn_list = []

        for el in edge_list:
            gg_el = self.graph_el.es[self.graph_el.get_eid(el[0], el[1])]
            el1 = gg_el["weight"]
            syn_list.append(el1)

        return syn_list

    def print_edge_prop(self, string_el):
        for i in self.graph_el.es():
            print(str(i[string_el]))

    def print_node_prop(self, string_el):
        for i in self.graph_el.vs():
            print(str(i[string_el]))

    def get_running_graph(self):
        if self.graph_el != 0:
            return self.graph_el
        else:
            return None

    def put_color_by_cell_class(self):
        list_class = []
        for i in self.graph_el.vs():
            if i["role"] == 'S':
                list_class.append(0)
            elif i["role"] == 'M':
                list_class.append(1)
            else:
                list_class.append(2)

        self.graph_el.vs["color"] = np.asarray(list_class)
        return self.graph_el, list_class


# summary(elegans_graph.put_color_by_cell_class())

# To characterize the direct impact that one neuron can have on
# another, we quantify the strength of connections by the
# multiplicity, m ij , between neurons i and j, which is the number
# of synaptic contacts (here gap junctions) connecting i to j. The
# degree treats synaptic connections as binary, whereas the
# multiplicity, also called edge weight, quantifies the number of contacts.
# elegans_graph.print_edge_prop("weight")
# elegans_graph.print_node_prop("color")
