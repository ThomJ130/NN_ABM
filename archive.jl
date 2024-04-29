
function linearLayout(shape::Vector{Int}; r=1.0, pos=nothing)
    # add dim option?    
    if isa(pos, Observable)
        pos = pos[]
    end

    function ll2d(g::AbstractGraph)
        if !isnothing(pos)
            spacing = reverse(pos)
            # println(pos)
        end
        xs = [p[1] for p in pos]
        ys = [p[2] for p in pos]
        # println("(layer,\tnode,\theight)")
        # for (layer, width) in enumerate(shape)
        #     # println("Layer: $layer, Width: $width")
        #     y_height = 0
        #     append!(spacing, 0.0)

        #     for y in 1:width
        #         print(spacing)
        #         y_height += !isnothing(spacing) ? pop!(spacing) : 1
        #         # println("($layer,\t$y,\t$y_height)")
        #         push!(ys, y_height * r)
        #         push!(xs, (layer - 1) * r)

        #         # println("idx = $((layer-1)+(y)), d = $(spacing[(layer-1)+(y)])")
        #     end
        #     println()
        # end
        return Point.(zip(xs, ys))
    end

    return ll2d
end


# g = smallgraph(:diamond)
# graphplot(g; layout=Stress(; dim=3), node_size=40)

# Training loop, using the whole data set 1000 times:
# losses = []
# @showprogress for epoch in 1:1_000
#     for (x, y) in loader
#         loss, grads = Flux.withgradient(network) do m
#             # Evaluate network and loss inside gradient context:
#             y_hat = m(x)
#             Flux.crossentropy(y_hat, y)
#         end
#         Flux.update!(optim, network, grads[1])
#         push!(losses, loss)  # logging, outside gradient context
#     end
# end

# optim # parameters, momenta and output have all changed
# out2 = network(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)

# mean((out2[1, :] .> 0.5) .== truth)  # accuracy 94% so far!
