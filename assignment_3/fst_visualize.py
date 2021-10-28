from graphviz import Digraph

if __name__ == "__main__":
    hfst_file = "fst.txt"
    txt_file = open(hfst_file, "r")

    # choose a subset of this before running, as there are too many lines to
    # visualize all
    lines = txt_file.readlines()
    Lines = lines[0:10]

    f = Digraph("finite_state_machine", filename="fst")
    f.attr(rankdir="LR", size="8,5")

    f.attr(
        "node",
        shape="circle",
        fontname="Tahoma",
        fontsize="14",
        fillcolor="grey",
        style="filled",
    )
    f.attr("edge", fontname="FreeMono", fontsize="14")

    # create the graph
    for line in Lines:
        line = line.strip()
        row = line.split("\t")
        if len(row) >= 4:
            f.node(row[0])
            f.node(row[1])
            f.edge(row[0], row[1], label=row[2] + ":" + row[3])
        elif len(row) == 2:
            f.attr("node", shape="doublecircle")
            f.node(row[0])

    # export graph to pdf
    f.view()
