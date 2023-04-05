using GraphViz, FileIO, ImageIO

function draw_tree(draw_tree_nodes)
    tree = """graph{layout="dot";"""
    drawn = []
    counter = 0
    or_counter = 10000
    parents = collect(keys(draw_tree_nodes))
    for (n, parent) in enumerate(parents)
        children = draw_tree_nodes[parent]
        chds = []
        for child in children
            if child[1] == 'm'
                tree *= """ "$(child)"[fixedsize=shape;shape=diamond;label="$(child[2:end])"fillcolor="lime";style=filled;height=.5;width=.5;];"""
            elseif child ∉ parents
                tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
            elseif n == length(draw_tree_nodes)
                tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
            else
                tree *= """ "$(child)"[label="$(child[1])"];"""
            end
            push!(drawn, child)
            push!(chds, string(child[1]))
        end
        if parent ∉ drawn
            tree *= """ "$(parent)"[label="$(parent[1])"];"""
            push!(drawn, parent)
        end
        childss = reverse(split(parent, "-")[2:end])
        counter3 = 1
        for childs in childss
            if length(childs) == 0
                for child in children
                    if child[1] == 'm'
                        #tree *= """ $(child) [shape=diamond,style=filled,label="",height=.1,width=.1];"""
                        tree *= """ "$(parent)"--"$(child)";"""
                        #tree *= """ "$(or_counter)"--"$(child)";"""
                        or_counter += 1
                        counter3 += 1
                    end
                end
                continue
            end
            tree *= """ $(or_counter) [shape=diamond,style=filled,label="",height=.3,width=.3];"""
            tree *= """ "$(parent)"--"$(or_counter)";"""
            childs = split(childs, "")
            for chld in childs
                if chld ∉ chds
                    tree *= """ "$(counter)"[label="$(chld)"fillcolor="red";style=filled;];"""
                    tree *= """ "$(or_counter)"--"$(counter)";"""
                    counter += 1
                else
                    tree *= """ "$(or_counter)"--"$(children[counter3])";"""
                    counter3 += 1
                end
            end
            or_counter += 1
        end
    end
    tree *= "}"
    return GraphViz.Graph(tree)
end
