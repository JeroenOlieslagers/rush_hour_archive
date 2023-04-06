using GraphViz, FileIO, ImageIO

function draw_tree(draw_tree_nodes)
    tree = """graph{layout="dot";"""
    drawn = []
    or_counter = 100000
    child_drawn_counter = 1000000
    parents = collect(keys(draw_tree_nodes))
    for (n, parent) in enumerate(parents)
        if parent ∉ drawn
            tree *= """ "$(parent)"[label="$(parent[1])"];"""
            push!(drawn, parent)
        end
        children = draw_tree_nodes[parent]
        chds = zeros(Int, length(children))
        for (m, child) in enumerate(children)
            if child in drawn
                tree *= """ "$(child_drawn_counter)"[label="$(child[child[1]=='v' ? 2 : 1])"fillcolor="red";style=filled;];"""
                chds[m] = child_drawn_counter
                child_drawn_counter += 1
            elseif child[1] == 'm'
                tree *= """ "$(child)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="lime";style=filled;height=.5;width=.5;];"""
            elseif child[1] == 'v'#child ∉ parents
                tree *= """ "$(child)"[label="$(child[2])"fillcolor="red";style=filled;];"""
            elseif n == length(draw_tree_nodes)
                tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
            else
                tree *= """ "$(child)"[label="$(child[1])"];"""
            end
            push!(drawn, child)
        end
        # if dfs
        #     childss = reverse(split(parent, "_")[2:end])
        # else
        childss = split(parent, "_")[2:end]
        # end
        counter3 = 1
        for childs in childss
            if children[counter3][1] == 'm'
                tree *= """ "$(parent)"--"$(children[counter3])";"""
                or_counter += 1
                counter3 += 1
                continue
            end
            tree *= """ $(or_counter) [shape=diamond,style=filled,label="",height=.3,width=.3];"""
            tree *= """ "$(parent)"--"$(or_counter)";"""
            # if dfs
            #     childs = reverse(split(childs, ""))
            # else
            childs = split(childs, "")
            # end
            # println(chds)
            # println(childs)
            # println("======")
            for chld in childs
                if chds[counter3] > 0
                    tree *= """ "$(or_counter)"--"$(chds[counter3])";"""
                else
                    tree *= """ "$(or_counter)"--"$(children[counter3])";"""
                end
                counter3 += 1
            end
            or_counter += 1
        end
    end
    tree *= "}"
    return GraphViz.Graph(tree)
end