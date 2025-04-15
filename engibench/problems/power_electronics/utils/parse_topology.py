"""Parse the topology from config.original_netlist_path."""
# ruff: noqa: N806, N815 # Upper case

import networkx as nx

from engibench.problems.power_electronics.utils.config import Config


def parse_topology(
    config: Config, components: dict[str, int] = {"V": 0, "R": 1, "C": 2, "S": 3, "L": 4, "D": 5}
) -> tuple[Config, nx.Graph]:
    """Parse the topology from config.original_netlist.Read from config.original_netlist_path to get the topology.

    It only keeps component lines for topology. So the following lines are discarded in this process: .PARAM, .model, .tran, .save etc.
    It also creates a nx.Graph object for the render() function.
    """
    config.cmp_edg_str = ""  # reset
    G = nx.Graph()  # reset
    color_dict: dict[str, str] = {"R": "b", "L": "g", "C": "r", "D": "yellow", "V": "orange", "S": "purple"}

    calc_comp_count = {"V": 0, "R": 0, "C": 0, "S": 0, "L": 0, "D": 0}
    ref_comp_count = {"V": 1, "R": 1, "C": config.n_C, "S": config.n_S, "L": config.n_L, "D": config.n_D}

    with open(config.original_netlist_path) as file:
        for line in file:
            if line.strip() != "":
                line_ = line.replace("=", " ")  # to deal with .PARAM problems. See the comments below.
                # liang: Note that line still contains \n at the end of it!
                line_spl = line_.split()
                if line_spl[0] == ".PARAM":
                    # e.g. .PARAM V0_value = 10
                    # e.g. .PARAM C0_value = 10u
                    # e.g. .PARAM L2_value=0.001. Note the whitespace! It appears in new files provided by RTRC.
                    # e.g. .PARAM GS2_L2=0
                    pass
                elif line[0] in components:
                    line_spl = line.split(" ")[:3]

                    if "RC" in line_spl[0] or "_GS" in line_spl[0]:
                        continue  # pass this line

                    config.edge_map[line_spl[0]] = [int(line_spl[1]), int(line_spl[2])]
                    config.cmp_edg_str = (
                        config.cmp_edg_str + line
                    )  # Liang: do not add another '\n' at the end of this line.
                    calc_comp_count[line[0]] = calc_comp_count[line[0]] + 1

                    G.add_node(line_spl[0], bipartite=0, color=color_dict[line[0]])
                    G.add_node(line_spl[1], bipartite=1, color="gray")
                    G.add_node(line_spl[2], bipartite=1, color="gray")
                    G.add_edge(line_spl[0], line_spl[1])
                    G.add_edge(line_spl[0], line_spl[2])
    assert ref_comp_count == calc_comp_count, "Error when parsing topology: component counts do not match!"
    return config, G
